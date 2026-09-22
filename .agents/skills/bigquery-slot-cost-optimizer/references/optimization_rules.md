# BigQuery query and architecture optimization heuristics

This reference catalog documents concrete optimization heuristics, architectural rules, and SQL rewrite patterns for Google Cloud BigQuery workloads. It is used by both automated diagnostic engines and human practitioners to resolve query bottlenecks and reduce query costs.

## Table of contents

- [Partitioning optimization](#partitioning-optimization): lines 15-116
- [Multi-column clustering optimization](#multi-column-clustering-optimization): lines 117-149
- [BI Engine in-memory acceleration](#bi-engine-in-memory-acceleration): lines 150-169
- [BigQuery search indexes](#bigquery-search-indexes): lines 170-203
- [Materialized views and transparent query rewriting](#materialized-views-and-transparent-query-rewriting): lines 204-230
- [Join optimization and data skew mitigation](#join-optimization-and-data-skew-mitigation): lines 231-290
- [Official BigQuery documentation links for progressive disclosure](#official-bigquery-documentation-links-for-progressive-disclosure): lines 291-301

## Partitioning optimization

Table partitioning divides large tables into smaller segments, significantly reducing bytes scanned and slot consumption by pruning unneeded partitions at query planning time.

### Partitioning types

1. **Time-unit column partitioning**:
   - Partitioned on `DATE`, `DATETIME`, or `TIMESTAMP` columns.
   - Granularities: Hour, Day, Month, Year.
   - Example DDL:

     ```sql
     CREATE OR REPLACE TABLE `<PROJECT_ID>.ecommerce.orders`
     (
       order_id STRING,
       customer_id STRING,
       order_timestamp TIMESTAMP,
       total_amount NUMERIC
     )
     PARTITION BY TIMESTAMP_TRUNC(order_timestamp, DAY)
     OPTIONS (
       require_partition_filter = TRUE,
       partition_expiration_days = 730
     );
     ```

1. **Ingestion-time partitioning**:
   - Uses pseudo-columns `_PARTITIONTIME` or `_PARTITIONDATE`.
   - Useful when tables lack natural timestamp fields.

1. **Integer range partitioning**:
   - Partitioned on an `INT64` column using `GENERATE_ARRAY(start, end, interval)`.
   - Example: Customer ID ranges or account segment IDs.

### Identifying unpartitioned scans and verifying table partitioning

To diagnose and remediate unpartitioned full table scans, always follow these 4 steps:

1. **Inspect both `total_bytes_billed` and `total_bytes_processed` in `INFORMATION_SCHEMA.JOBS_BY_PROJECT`**:
   Query `INFORMATION_SCHEMA.JOBS_BY_PROJECT` selecting both `total_bytes_billed` and `total_bytes_processed` to identify high-cost queries scanning full tables:

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

2. **Verify whether referenced tables have date, timestamp, or integer-range partitioning**:
   Query `INFORMATION_SCHEMA.COLUMNS` (checking `is_partitioning_column = 'YES'` and inspecting `data_type` for `DATE`, `TIMESTAMP`, `DATETIME`, or `INT64` integer-range partitioning) and `INFORMATION_SCHEMA.PARTITIONS` to confirm whether referenced tables are partitioned:

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

3. **Enforce partition filters (`require_partition_filter = TRUE`)**:
   Enable `require_partition_filter = TRUE` on large partitioned tables (>10 GB) to block accidental full table scans:

   ```sql
   ALTER TABLE `<PROJECT_ID>.<DATASET>.orders`
   SET OPTIONS (require_partition_filter = TRUE);
   ```

4. **Combine partitioning with clustering on high-cardinality filtering and grouping columns**:
   Always recommend clustering (`CLUSTER BY`) on high-cardinality columns frequently used in `WHERE` filters, `JOIN` keys, and `GROUP BY` aggregations (up to 4 columns) alongside partitioning.

### Architectural guardrails

- **Max partitions**: a table cannot exceed 4,000 partitions. Daily partitioning supports approximately 10 years of historical data; hourly partitioning supports approximately 5.5 months.
- **Enforcing filters**: always set `OPTIONS (require_partition_filter = TRUE)` on high-volume tables (>100 GB) to prevent inadvertent full-table scans by ad-hoc queries.

### SQL antipatterns and rewrites

- **Antipattern: function on partition column**: applying a conversion function to a partitioned column invalidates partition pruning.

  ```sql
  -- BAD: Prevents partition pruning, scans entire table!
  SELECT COUNT(*)
  FROM `<PROJECT_ID>.ecommerce.orders`
  WHERE DATE(order_timestamp) = '2026-03-01';

  -- GOOD: Prunes partitions deterministically using timestamp bounds
  SELECT COUNT(*)
  FROM `<PROJECT_ID>.ecommerce.orders`
  WHERE order_timestamp >= '2026-03-01 00:00:00 UTC'
    AND order_timestamp < '2026-03-02 00:00:00 UTC';
  ```

## Multi-column clustering optimization

Clustering sorts data based on the contents of up to four columns, co-locating related data within storage blocks. BigQuery utilizes metadata min/max values to skip unneeded storage blocks during scans.

### Column ordering hierarchy

When defining clustered columns, ordering matters:

1. **Primary key or frequent equality filter**: place the most commonly filtered column first.
1. **High-cardinality dimension**: place dimensions frequently filtered together or used in inequality filters.
1. **Grouping or joining key**: place columns frequently used in `GROUP BY` or `JOIN` conditions.

Example DDL:

```sql
CREATE OR REPLACE TABLE `<PROJECT_ID>.ecommerce.orders_clustered`
(
  order_id STRING,
  customer_id STRING,
  region_id STRING,
  order_timestamp TIMESTAMP,
  total_amount NUMERIC
)
PARTITION BY DATE(order_timestamp)
CLUSTER BY customer_id, region_id;
```

### Synergistic partitioning and clustering

- BigQuery evaluates partition bounds first, pruning entire partition storage directories.
- Within matching partitions, BigQuery evaluates clustering min/max block metadata, scanning only matching 100MB-1GB blocks.
- This combination regularly yields 90-99% reductions in scanned bytes and slot execution time.

## BI Engine in-memory acceleration

Google Cloud BigQuery BI Engine is a built-in, distributed in-memory analysis service that accelerates SQL queries in real time with sub-second latency and zero slot usage from reservations.

### Qualifying queries for in-memory execution

- Suitable for repetitive analytical dashboards (Looker, Looker Studio, Tableau) and interactive SQL aggregations.
- Inspect execution plans in `INFORMATION_SCHEMA.JOBS_BY_*` for:
  - `query_info.bi_engine_statistics.bi_engine_mode = 'FULL'` (Fully accelerated).
  - `query_info.bi_engine_statistics.bi_engine_mode = 'PARTIAL'` (Partially accelerated; some stages ran in slots).
  - `query_info.bi_engine_statistics.bi_engine_mode = 'DISABLED'` (BI Engine skipped).

### Unsupported BI Engine constructs

If queries fall back from BI Engine to standard slot compute, inspect `bi_engine_reasons`:
- Queries containing non-deterministic functions (e.g., `CURRENT_TIMESTAMP()`, `RAND()`).
- Complex JavaScript User-Defined Functions (UDFs).
- Non-equi joins (`ON a.val > b.val`).
- Tables exceeding BI Engine reservation memory capacity without partitioning.

## BigQuery search indexes

Search indexing provides sub-second point lookups and substring search across high-volume log, telemetry, and unstructured JSON datasets without scanning billions of rows.

### Index creation

Create search indexes across all columns or targeted string/JSON columns:

```sql
-- Index all text columns in audit log table
CREATE SEARCH INDEX IF NOT EXISTS logs_search_idx
ON `<PROJECT_ID>.telemetry.application_logs`(ALL COLUMNS);

-- Index targeted JSON payload column
CREATE SEARCH INDEX IF NOT EXISTS payload_search_idx
ON `<PROJECT_ID>.telemetry.events`(payload);
```

### Query patterns and rewrites

- Use the built-in `SEARCH()` function to leverage the inverted B-tree index.

```sql
-- BAD: Full table scan scanning hundreds of gigabytes
SELECT timestamp, log_level, message
FROM `<PROJECT_ID>.telemetry.application_logs`
WHERE REGEXP_CONTAINS(message, r'FATAL_EXCEPTION_500');

-- GOOD: Search index lookup scanning near zero bytes
SELECT timestamp, log_level, message
FROM `<PROJECT_ID>.telemetry.application_logs`
WHERE SEARCH(message, '`FATAL_EXCEPTION_500`');
```

## Materialized views and transparent query rewriting

Materialized views periodically pre-compute and store aggregated result sets with automatic incremental maintenance and zero administrative pipeline overhead.

### Automatic query rewrite engine

A major architectural advantage in BigQuery is transparent query rewriting:
- Even if a user query targets the raw base table, the BigQuery optimizer inspects matching materialized views.
- If the view covers the requested dimensions and metrics, BigQuery automatically reroutes the query to scan the pre-computed materialized view!
- **Partition alignment**: always align the partitioning column and granularity of a materialized view with its underlying base table (e.g., `PARTITION BY DATE(order_timestamp)`) to maximize incremental refresh efficiency and avoid full view recomputation.

Example DDL:

```sql
CREATE MATERIALIZED VIEW `<PROJECT_ID>.ecommerce.mv_daily_sales_by_region`
PARTITION BY order_date
CLUSTER BY region_id
AS
SELECT
  DATE(order_timestamp) AS order_date,
  region_id,
  COUNT(order_id) AS total_orders,
  SUM(total_amount) AS total_revenue
FROM `<PROJECT_ID>.ecommerce.orders`
GROUP BY 1, 2;
```

## Join optimization and data skew mitigation

Inefficient joins are the primary cause of memory exhaustion, shuffle disk spillage, and explosive slot-hour consumption.

### Hash join compared to broadcast join

- In a **Broadcast Join**, BigQuery sends the entire right table to each slot processing the left table.
- Best practice: always place the largest table on the LEFT side of the `JOIN` keyword and the smaller dimension table on the RIGHT side.
- BigQuery automatically chooses broadcast join when the right table is under ~50MB.

### Cartesian join elimination

- A Cartesian product (`CROSS JOIN`, missing `ON` condition, or non-selective/many-to-many join predicate) multiplies row counts exponentially ($M \times N$), causing memory exhaustion and shuffle spilling to persistent disk.
- **Mandatory 4-step diagnostic and remediation checklist**:
  1. **Check stage telemetry for row count explosions**: query `INFORMATION_SCHEMA.JOBS_BY_PROJECT` (`job_stages`) or inspect the query execution graph to identify stages where output rows (`records_written`) drastically exceed input rows (`records_read`).
  2. **Check for missing or non-selective join predicates**: inspect the SQL query text for missing `ON` conditions, unintentional `CROSS JOIN` syntax, or non-selective predicates (`ON 1=1`), in addition to checking for duplicate keys across joined tables.
  3. **Check for memory spillage to persistent storage**: verify whether `shuffle_output_bytes_spilled > 0` in stage telemetry, indicating that intermediate join state overwhelmed slot memory buffers and spilled to disk.
  4. **Pre-aggregate dimensional data or enforce distinct keys**: pre-aggregate dimensional or activity tables down to unique join keys in CTEs before joining, or enforce `DISTINCT` key constraints to eliminate row multiplication:

```sql
-- BAD: Exploding Cartesian join due to missing ON condition, unintentional CROSS JOIN, or duplicate activity keys
SELECT
  c.customer_id,
  t.transaction_amount,
  e.event_name
FROM `<PROJECT_ID>.ecommerce.customers` c
JOIN `<PROJECT_ID>.ecommerce.transactions` t ON c.customer_id = t.customer_id
JOIN `<PROJECT_ID>.ecommerce.web_events` e ON c.customer_id = e.customer_id;

-- GOOD: Pre-aggregated metrics joined cleanly without row multiplication
WITH txn_summary AS (
  SELECT customer_id, SUM(transaction_amount) AS total_spent
  FROM `<PROJECT_ID>.ecommerce.transactions`
  GROUP BY customer_id
),
event_summary AS (
  SELECT customer_id, COUNT(*) AS total_events
  FROM `<PROJECT_ID>.ecommerce.web_events`
  GROUP BY customer_id
)
SELECT
  c.customer_id,
  COALESCE(t.total_spent, 0) AS total_spent,
  COALESCE(e.total_events, 0) AS total_events
FROM `<PROJECT_ID>.ecommerce.customers` c
LEFT JOIN txn_summary t ON c.customer_id = t.customer_id
LEFT JOIN event_summary e ON c.customer_id = e.customer_id;
```

### Mitigating data skew

When a join key has heavy key concentration (e.g. `customer_id IS NULL` or `customer_id = 'DEFAULT'`), a single slot receives millions of rows while other slots remain idle.
- Filter out dummy/NULL keys before joining:

  ```sql
  WHERE customer_id IS NOT NULL AND customer_id != 'GUEST'
  ```

- Salting: for skewed non-null keys, append a random integer hash `MOD(FARM_FINGERPRINT(id), 10)` to distribute across slots.

## Official BigQuery documentation links for progressive disclosure

To optimize token usage and ensure access to the latest syntax and limits without loading all documentation at once, load the following official Google Cloud markdown (`.md.txt`) documentation pages on demand when deeper details are required:

- **Partitioned tables (`https://docs.cloud.google.com/bigquery/docs/partitioned-tables.md.txt`)**: load when designing partition expiration policies, integer-range partitioning boundaries, or troubleshooting partition pruning edge cases.
- **Clustered tables (`https://docs.cloud.google.com/bigquery/docs/clustered-tables.md.txt`)**: load when evaluating automatic re-clustering behavior or multi-column block pruning mechanics.
- **BI Engine (`https://docs.cloud.google.com/bigquery/docs/bi-engine-intro.md.txt`)**: load when checking current supported SQL functions, operators, and reservation capacity limits for in-memory acceleration.
- **Search indexes (`https://docs.cloud.google.com/bigquery/docs/search-intro.md.txt`)**: load when configuring `CREATE SEARCH INDEX` analyzer options (`LOG_ANALYZER`, `NO_OP_ANALYZER`) or querying nested JSON fields with `SEARCH()`.
- **Materialized views (`https://docs.cloud.google.com/bigquery/docs/materialized-views-intro.md.txt`)**: load when verifying incremental refresh eligibility rules or smart tuning query rewrite constraints.
- **Compute performance best practices (`https://docs.cloud.google.com/bigquery/docs/best-practices-performance-compute.md.txt`)**: load when resolving complex distributed join skew, broadcast join sizing limits, or shuffle spillage.
