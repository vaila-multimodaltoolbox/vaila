# Query Performance Queries

Telemetry and SQL queries for evaluating individual and aggregate job
performance, discovering comparable job executions, analyzing slot consumption,
identifying query/stage bottlenecks, inspecting table-level job activity, and
verifying BI Engine / metadata cache (cmeta) acceleration.

## Table of Contents

-   [Single Job Performance & Stage Bottleneck Flags](#single-job-performance-stage-bottleneck-flags)
    (Lines 22-71)
-   [REST API Single-Job Point-Lookup & Stage Bottlenecks](#rest-api-single-job-point-lookup-stage-bottlenecks)
    (Lines 72-92)
-   [Finding Comparable Jobs](#finding-comparable-jobs) (Lines 93-124)
-   [BI Engine Acceleration Usage](#bi-engine-acceleration-usage) (Lines
    125-147)
-   [Table Metadata Cache (cmeta) Usage](#table-metadata-cache-cmeta-usage)
    (Lines 148-174)
-   [Query Performance Variance & Outlier Discovery](#query-performance-variance-outlier-discovery)
    (Lines 175-208)

## Single Job Performance & Stage Bottleneck Flags

Retrieve execution duration, queuing delay, compute usage, bytes processed,
historical average duration, and stage bottleneck flags for a specific job.

```sql
SELECT
  job_id,
  user_email,
  state,
  creation_time,
  start_time,
  end_time,
  TIMESTAMP_DIFF(end_time, start_time, MILLISECOND) AS duration_ms,
  TIMESTAMP_DIFF(start_time, creation_time, MILLISECOND) AS pending_ms,
  total_slot_ms,
  ROUND(
    SAFE_DIVIDE(
      total_slot_ms, TIMESTAMP_DIFF(end_time, start_time, MILLISECOND)),
    1) AS avg_slots_used,
  total_bytes_processed,
  total_bytes_billed,
  reservation_id,
  query_info.performance_insights.avg_previous_execution_ms
    AS avg_previous_execution_ms,
  query_info.performance_insights.avg_previous_slot_ms AS avg_previous_slot_ms,
  query_info.performance_insights.avg_previous_bytes_processed
    AS avg_previous_bytes_processed,
  query_info.performance_insights.num_previous_successful_jobs
    AS num_previous_successful_jobs,
  (
    SELECT LOGICAL_OR(slot_contention)
    FROM
      UNNEST(
        query_info.performance_insights.stage_performance_standalone_insights)
  ) AS has_slot_contention,
  (
    SELECT LOGICAL_OR(insufficient_shuffle_quota)
    FROM
      UNNEST(
        query_info.performance_insights.stage_performance_standalone_insights)
  ) AS has_insufficient_shuffle_quota,
  error_result.message AS error_message
FROM
  `{project_id}`.`region-{region}`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE
  creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  AND job_id = '{job_id}';
```

## REST API Single-Job Point-Lookup & Stage Bottlenecks

Use the REST API for fast, zero-SQL point-lookups to extract stage-level
execution graphs, wait ratios, and performance insights.

```bash
bq show --location={location} -j {project_id}:{job_id}
```

Key fields in JSON response:

-   `statistics.query.performanceInsights`: Execution bottleneck diagnostics
    (e.g., `stagePerformanceStandaloneInsights[].slotContention`,
    `stagePerformanceStandaloneInsights[].insufficientShuffleQuota`,
    `stagePerformanceChangeInsights[].inputDataChange`,
    `avgPreviousExecutionMs`).
-   `statistics.query.queryPlan[]`: Stage-by-stage stats including `slotMs`,
    `shuffleOutputBytes`, `recordsRead`, `recordsWritten`, `computeRatioAvg`,
    `computeRatioMax`, `waitRatioAvg`, `waitRatioMax`, `readRatioAvg`,
    `readRatioMax`, `writeRatioAvg`, `writeRatioMax`.

## Finding Comparable Jobs

Use `query_info.query_hashes.normalized_literals` to discover past runs of the
same query template and rank by duration to find fast baseline executions.

```sql
SELECT
  job_id,
  creation_time,
  TIMESTAMP_DIFF(end_time, start_time, MILLISECOND) AS duration_ms,
  total_slot_ms,
  ROUND(SAFE_DIVIDE(total_slot_ms, TIMESTAMP_DIFF(end_time, start_time, MILLISECOND)), 1)
    AS avg_slots_used,
  total_bytes_processed,
  total_bytes_billed,
  reservation_id,
  (
    SELECT LOGICAL_OR(slot_contention)
    FROM UNNEST(query_info.performance_insights.stage_performance_standalone_insights)
  ) AS has_slot_contention
FROM
  `{project_id}`.`region-{region}`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE
  creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 14 DAY)
  AND (statement_type != 'SCRIPT' OR statement_type IS NULL)
  AND state = 'DONE'
  AND query_info.query_hashes.normalized_literals = '{target_normalized_hash}'
ORDER BY
  duration_ms ASC
LIMIT 20;
```

## BI Engine Acceleration Usage

Verify whether queries leveraged BI Engine memory acceleration and diagnose
acceleration rejection reasons.

```sql
SELECT
  job_id,
  creation_time,
  total_slot_ms,
  bi_engine_statistics.bi_engine_mode,
  bi_engine_statistics.acceleration_mode,
  bi_engine_statistics.bi_engine_reasons
FROM
  `{project_id}`.`region-{region}`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE
  creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 3 DAY)
  AND bi_engine_statistics IS NOT NULL
ORDER BY
  creation_time DESC
LIMIT 50;
```

## Table Metadata Cache (cmeta) Usage

Inspect whether queries leveraged table metadata caching (`cmeta`), check
staleness, and diagnose why metadata caching was unused.

```sql
SELECT
  job_id,
  creation_time,
  total_slot_ms,
  t.table_reference.project_id AS table_project_id,
  t.table_reference.dataset_id AS table_dataset_id,
  t.table_reference.table_id,
  t.unused_reason,
  t.explanation,
  t.staleness_seconds,
FROM
  `{project_id}`.`region-{region}`.INFORMATION_SCHEMA.JOBS_BY_PROJECT,
  UNNEST(metadata_cache_statistics.table_metadata_cache_usage) AS t
WHERE
  creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 3 DAY)
  AND (statement_type != 'SCRIPT' OR statement_type IS NULL)
ORDER BY
  creation_time DESC
LIMIT 50;
```

## Query Performance Variance & Outlier Discovery

Identify queries that experienced the highest execution variance compared to
their historical baseline duration (`avg_previous_execution_ms`).

```sql
SELECT
  job_id,
  user_email,
  reservation_id,
  creation_time,
  ROUND(TIMESTAMP_DIFF(end_time, start_time, MILLISECOND) / 1000.0, 1) AS duration_sec,
  ROUND(query_info.performance_insights.avg_previous_execution_ms / 1000.0, 1)
    AS baseline_avg_duration_sec,
  ROUND(
    SAFE_DIVIDE(
      TIMESTAMP_DIFF(end_time, start_time, MILLISECOND),
      query_info.performance_insights.avg_previous_execution_ms),
    2) AS variance_ratio,
  total_slot_ms,
  total_bytes_billed
FROM
  `{project_id}`.`region-{region}`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE
  creation_time
    BETWEEN TIMESTAMP('{target_start_timestamp}')
    AND TIMESTAMP('{target_end_timestamp}')
  AND state = 'DONE'
  AND (statement_type != 'SCRIPT' OR statement_type IS NULL)
  AND query_info.performance_insights.avg_previous_execution_ms IS NOT NULL
ORDER BY
  variance_ratio DESC
LIMIT 50;
```
