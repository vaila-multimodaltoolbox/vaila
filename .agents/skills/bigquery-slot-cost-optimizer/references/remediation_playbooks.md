# BigQuery concrete remediation playbooks

This reference document contains the mandatory diagnostic checklists, root-cause analysis, and SQL remediation playbooks for `Rule SLOT-001`, `Rule JOIN-001`, and `Rule PART-001`. Agents diagnosing BigQuery performance or cost bottlenecks must follow all steps in the applicable playbook.

## Table of contents

- [Rule SLOT-001: slot contention and queueing](#rule-slot-001-slot-contention-and-queueing): lines 11-43
- [Rule JOIN-001: Cartesian and exploding joins](#rule-join-001-cartesian-and-exploding-joins): lines 44-93
- [Rule PART-001: unpartitioned scans and partition pruning](#rule-part-001-unpartitioned-scans-and-partition-pruning): lines 94-154

## Rule SLOT-001: slot contention and queueing

- **Symptoms**: stages have high `wait_ratio_avg` (> 40%), `total_slot_ms` is high, but wall-clock time is disproportionately prolonged.
- **Root cause**: the query is competing for slots in an oversubscribed on-demand pool or undersized capacity reservation.
- **Mandatory diagnostic and remediation workflow (include all 4 steps in your analysis)**:
  1. **Query `INFORMATION_SCHEMA.JOBS_BY_PROJECT` or `JOBS_TIMELINE_BY_*`**: inspect `total_slot_ms`, `job_stages` (`wait_ratio_avg`), and concurrent slot utilization over time.
  2. **Calculate slot-hours and analyze concurrent slot demands**: compute total slot-hours (`total_slot_ms / (1000 * 60 * 60)`) and evaluate peak concurrent slot demand against available baseline slot capacity.
  3. **Identify queueing stages or slot contention**: inspect stage execution telemetry where `wait_ratio_avg > 0.40` or `EXISTS(SELECT 1 FROM UNNEST(query_info.performance_insights.stage_performance_standalone_insights) WHERE slot_contention)`.
  4. **Provide actionable mitigations**:
     - **Allocate capacity reservations or configure slot autoscaling**: in BigQuery Editions (`Standard`, `Enterprise`, `Enterprise Plus`), allocate dedicated baseline slots with an autoscaling ceiling to accommodate burst workloads.
     - **Reduce peak query concurrency**: stagger scheduled batch ETL queries that launch simultaneously at midnight UTC and isolate ad-hoc BI queries into separate reservations.
     - **Optimize heavy shuffle stages**: reduce intermediate data volume in stages with massive row shuffles to shorten slot holding duration.

  - **Diagnostic SQL for slot utilization and stage queueing**:

    ```sql
    SELECT
      job_id,
      user_email,
      total_slot_ms / (1000 * 60 * 60) AS slot_hours,
      TIMESTAMP_DIFF(end_time, start_time, SECOND) AS elapsed_seconds,
      EXISTS(
        SELECT 1
        FROM UNNEST(query_info.performance_insights.stage_performance_standalone_insights)
        WHERE slot_contention
      ) AS slot_contention
    FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
    WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
    ORDER BY total_slot_ms DESC
    LIMIT 20;
    ```

## Rule JOIN-001: Cartesian and exploding joins

- **Symptoms**: query execution stage telemetry shows massive row count explosions where `records_written` drastically exceeds `records_read` by orders of magnitude, accompanied by memory spillage to persistent storage (`shuffle_output_bytes_spilled` > 0).
- **Root cause**: missing or non-selective join predicates (such as unintentional `CROSS JOIN`, missing `ON` conditions, or tautological `ON 1=1` predicates) or non-unique many-to-many join keys causing duplicate row generation ($M \times N$ expansion).
- **Mandatory diagnostic and remediation workflow (include all 4 steps in your analysis)**:
  1. **Check stage telemetry for row count explosions**: query `INFORMATION_SCHEMA.JOBS_BY_PROJECT` (`job_stages`) or inspect the execution graph to identify stages where output rows (`records_written`) drastically exceed input rows (`records_read`).
  2. **Check for missing or non-selective join predicates**: explicitly inspect every `JOIN` clause in the SQL query text for missing `ON` conditions, unintentional `CROSS JOIN` syntax, or non-selective join predicates (such as `ON 1=1`), in addition to checking for duplicate keys across joined tables.
  3. **Check for memory spillage to persistent storage**: check stage telemetry for `shuffle_output_bytes_spilled > 0` (shuffle disk spillage caused by intermediate join state exceeding slot memory buffers).
  4. **Pre-aggregate dimensional data or enforce distinct keys**: pre-aggregate dimensional/activity tables down to unique join keys in CTEs before joining, or enforce `DISTINCT` key constraints to eliminate row multiplication:

  - **Diagnostic SQL for stage row explosion and shuffle spillage**:

    ```sql
    SELECT
      job_id,
      stage.name AS stage_name,
      stage.records_read,
      stage.records_written,
      SAFE_DIVIDE(stage.records_written, NULLIF(stage.records_read, 0)) AS row_expansion_ratio,
      stage.shuffle_output_bytes_spilled
    FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT,
    UNNEST(job_stages) AS stage
    WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
      AND (stage.records_written > stage.records_read * 10 OR stage.shuffle_output_bytes_spilled > 0)
    ORDER BY stage.shuffle_output_bytes_spilled DESC;
    ```

  - **Antipattern (unintentional CROSS JOIN / missing ON condition)**:

    ```sql
    SELECT *
    FROM `orders` o
    CROSS JOIN `web_events` e
    WHERE o.customer_id = e.customer_id;
    ```

  - **Optimized SQL (qualified equi-join + pre-aggregation CTE)**:

    ```sql
    WITH agg_events AS (
      SELECT customer_id, COUNT(*) AS event_count
      FROM `web_events`
      GROUP BY customer_id
    )
    SELECT o.order_id, o.customer_id, e.event_count
    FROM `orders` o
    INNER JOIN agg_events e
      ON o.customer_id = e.customer_id;
    ```

## Rule PART-001: unpartitioned scans and partition pruning

- **Symptoms**: high `total_bytes_billed` and `total_bytes_processed` (> 10 GB) in `INFORMATION_SCHEMA.JOBS_BY_PROJECT` when scanning historical logs or transaction history.
- **Root cause**: table lacks partitioning, query omits partition filter predicates, or query wraps partitioned columns in functions that prevent partition pruning.
- **Mandatory diagnostic and remediation workflow (include all 4 steps in your analysis)**:
  1. **Inspect both `total_bytes_billed` and `total_bytes_processed` in `INFORMATION_SCHEMA.JOBS_BY_PROJECT`**: run a diagnostic query selecting **both** `total_bytes_billed` and `total_bytes_processed` to identify expensive full table scans:

     ```sql
     SELECT
       job_id,
       user_email,
       total_bytes_processed,
       total_bytes_billed,
       query
     FROM `region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
     WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
       AND total_bytes_processed > 10 * 1024 * 1024 * 1024
     ORDER BY total_bytes_billed DESC;
     ```

  2. **Verify date, timestamp, or integer-range partitioning on referenced tables**: query `INFORMATION_SCHEMA.COLUMNS` (checking `is_partitioning_column = 'YES'` and `data_type` for `DATE`, `TIMESTAMP`, `DATETIME`, or `INT64` integer-range partitioning) and `INFORMATION_SCHEMA.PARTITIONS` to verify whether referenced tables are partitioned and inspect their partition scheme:

     ```sql
     SELECT
       table_name,
       column_name,
       data_type AS partition_type,
       is_partitioning_column,
       clustering_ordinal_position
     FROM `<PROJECT_ID>.<DATASET>`.INFORMATION_SCHEMA.COLUMNS
     WHERE is_partitioning_column = 'YES'
        OR clustering_ordinal_position IS NOT NULL;
     ```

  3. **Enforce partition filters (`require_partition_filter = TRUE`)**: enable `require_partition_filter = TRUE` on large partitioned tables via `ALTER TABLE` or `CREATE TABLE` DDL to block accidental full table scans:

     ```sql
     ALTER TABLE `<PROJECT_ID>.<DATASET>.orders`
     SET OPTIONS (require_partition_filter = TRUE);
     ```

  4. **Recommend clustering on high-cardinality filtering and grouping columns**: always combine partitioning with multi-column clustering (`CLUSTER BY`) on high-cardinality columns frequently used in `WHERE` filters, `JOIN` keys, and `GROUP BY` clauses (up to 4 columns):

     ```sql
     CREATE OR REPLACE TABLE `<PROJECT_ID>.<DATASET>.orders_optimized`
     PARTITION BY DATE(order_timestamp)
     CLUSTER BY customer_id, region_id
     OPTIONS (require_partition_filter = TRUE)
     AS SELECT * FROM `<PROJECT_ID>.<DATASET>.orders`;
     ```

  - **Avoid function wrappers on partition columns**:

    ```sql
    -- BAD: Scans entire table because function wraps partitioned column
    WHERE DATE(order_timestamp) = '2026-03-01';

    -- GOOD: Enables constant partition pruning
    WHERE order_timestamp >= '2026-03-01 00:00:00 UTC'
      AND order_timestamp < '2026-03-02 00:00:00 UTC';
    ```
