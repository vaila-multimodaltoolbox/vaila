# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BigQuery Slot and Cost Optimizer CLI Utility.

Analyzes Google Cloud BigQuery INFORMATION_SCHEMA job telemetry to compute
slot-hours, identify slot contention, pinpoint Cartesian explosions,
and flag costly unpartitioned table scans.

Note:
    Pricing calculations for on-demand queries and editions slot-hours require
    live billing rates passed at runtime via --ondemand-rate and --slot-hour-rate.
    Actual Google Cloud BigQuery costs vary depending on region, chosen edition
    (Standard, Enterprise, Enterprise Plus), and baseline or commitment terms.
    Consult official pricing documentation: https://cloud.google.com/bigquery/pricing
"""

import argparse
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import io
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


# ==============================================================================
# Domain Models & Data Structures
# ==============================================================================


@dataclass
class JobStage:
    """Represents an individual execution stage of a BigQuery query job."""

    stage_id: int
    name: str
    records_read: int
    records_written: int
    shuffle_output_bytes: int
    shuffle_output_bytes_spilled: int
    wait_ratio_avg: float
    slot_ms: int


@dataclass
class PerformanceInsights:
    """Performance diagnostic insights extracted from query_info."""

    slot_contention: bool = False
    insufficient_shuffle_quota: bool = False
    high_cardinality_joins: bool = False
    partition_skew: bool = False


@dataclass
class QueryJobMetrics:
    """Computed metrics and diagnostic findings for an individual query job."""

    job_id: str
    project_id: str
    user_email: str
    start_time: datetime
    end_time: datetime
    query_text: str
    total_slot_ms: int
    total_bytes_billed: int
    cache_hit: bool
    slot_hours: float
    avg_slot_concurrency: float
    estimated_cost_usd_ondemand: float
    estimated_cost_usd_editions: float
    stages: List[JobStage] = field(default_factory=list)
    insights: PerformanceInsights = field(default_factory=PerformanceInsights)
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class AnalysisSummary:
    """Aggregated project-wide telemetry summary."""

    project_id: str
    region: str
    lookback_days: int
    total_jobs_analyzed: int
    total_slot_hours_consumed: float
    total_cost_usd_ondemand: float
    total_cost_usd_editions: float
    contention_job_count: int
    cartesian_job_count: int
    unpartitioned_job_count: int
    top_heavy_jobs: List[QueryJobMetrics] = field(default_factory=list)
    all_recommendations: List[Dict[str, str]] = field(default_factory=list)


# ==============================================================================
# Helper Math & Normalization Functions
# ==============================================================================


def normalize_region(region: str) -> str:
    """Normalizes regional strings to INFORMATION_SCHEMA qualifiers.

    Args:
        region: Input region name (e.g. 'us', 'eu', 'us-central1', 'region-us').

    Returns:
        Canonical regional qualifier (e.g. 'region-us', 'region-us-central1').
    """
    cleaned = region.strip().lower()
    if cleaned.startswith("region-"):
        return cleaned
    return f"region-{cleaned}"


def calculate_slot_hours(total_slot_ms: int) -> float:
    """Converts total slot milliseconds into slot-hours.

    Args:
        total_slot_ms: Total slot milliseconds consumed.

    Returns:
        Floating point slot-hours rounded to 4 decimal places.
    """
    if total_slot_ms <= 0:
        return 0.0
    return round(total_slot_ms / 3_600_000.0, 4)


def calculate_avg_slot_concurrency(
    total_slot_ms: int, start_time: datetime, end_time: datetime
) -> float:
    """Calculates average slot concurrency over the job's elapsed wall-clock time.

    Args:
        total_slot_ms: Total compute time across all slots in milliseconds.
        start_time: Query execution start timestamp.
        end_time: Query execution completion timestamp.

    Returns:
        Average number of concurrent slots utilized during execution.
    """
    if total_slot_ms <= 0:
        return 0.0
    elapsed_seconds = (end_time - start_time).total_seconds()
    elapsed_ms = max(1.0, elapsed_seconds * 1000.0)
    return round(total_slot_ms / elapsed_ms, 2)


def calculate_cost_estimates(
    total_bytes_billed: int,
    slot_hours: float,
    ondemand_rate_per_tib: float,
    editions_rate_per_slot_hour: float,
) -> Tuple[float, float]:
    """Computes dollar cost estimates for both On-Demand and Editions pricing.

    Note:
        AI agents and users must retrieve live regional pricing rates at runtime
        from https://cloud.google.com/bigquery/pricing (matching region, edition,
        and commitment tier) and pass them explicitly via --ondemand-rate and
        --slot-hour-rate.

    Args:
        total_bytes_billed: Total billed bytes scanned.
        slot_hours: Total slot-hours consumed.
        ondemand_rate_per_tib: Pricing rate in USD per TiB scanned.
        editions_rate_per_slot_hour: Pricing rate in USD per slot-hour.

    Returns:
        Tuple of (cost_ondemand_usd, cost_editions_usd).
    """
    bytes_per_tib = 1_099_511_627_776.0
    ondemand_usd = (total_bytes_billed / bytes_per_tib) * ondemand_rate_per_tib
    editions_usd = slot_hours * editions_rate_per_slot_hour
    return (round(ondemand_usd, 4), round(editions_usd, 4))


# ==============================================================================
# Detection Heuristics
# ==============================================================================

_CROSS_JOIN_PATTERN = re.compile(r"(?i)\bCROSS\s+JOIN\b")
_TAUTOLOGY_JOIN_PATTERN = re.compile(
    r"(?is)\bJOIN\b.+?\bON\s+(?:1\s*=\s*1|TRUE)\b"
)
_WHERE_CLAUSE_PATTERN = re.compile(r"(?i)\bWHERE\b")
_FUNCTION_WRAPPER_ANTIPATTERN = re.compile(
    r"(?i)\bWHERE\b[\s\S]*?\b(?:"
    r"(?:DATE|TIMESTAMP|DATETIME)\s*\(\s*[a-zA-Z_][a-zA-Z0-9_.]*\s*(?:,[^)]*)?\)"
    r"|EXTRACT\s*\(\s*[a-zA-Z_]+\s+FROM\s+[a-zA-Z_][a-zA-Z0-9_.]*\s*\)"
    r")\s*(?:=|<|>|<=|>=|\bBETWEEN\b|\bIN\b)"
)
_VALID_PARTITION_FILTER_PATTERN = re.compile(
    r"(?i)\bWHERE\b[\s\S]*?(?:"
    r"\b(?:_PARTITIONDATE|_PARTITIONTIME)\b"
    r"|\b(?:[a-zA-Z_][a-zA-Z0-9_.]*)?(?:date|time|ts|day|partition|created|updated)[a-zA-Z0-9_]*\b\s*(?:=|<|>|<=|>=|\bBETWEEN\b|\bIN\b)"
    r"|(?:=|<|>|<=|>=|\bBETWEEN\b)\s*(?:DATE|TIMESTAMP|DATETIME|CURRENT_DATE|CURRENT_TIMESTAMP|TIMESTAMP_SUB|DATE_SUB|TIMESTAMP_TRUNC|DATE_TRUNC|['\"]\d{4}-\d{2}-\d{2})"
    r")"
)


def detect_cartesian_joins(
    stages: List[JobStage], query_text: str
) -> Tuple[bool, List[str]]:
    """Detects Cartesian joins, row count blowups, and shuffle disk spills.

    Args:
        stages: List of query execution stages.
        query_text: Raw SQL text of the query.

    Returns:
        Tuple of (is_cartesian, list_of_reasons).
    """
    reasons: List[str] = []

    # Check stages for row explosion or disk spilling
    for stage in stages:
        if stage.records_read > 1_000 and stage.records_written > (
            10 * stage.records_read
        ):
            ratio = (
                stage.records_written / stage.records_read
                if stage.records_read > 0
                else 0
            )
            reasons.append(
                f"Stage '{stage.name}' suffered {ratio:.1f}x row explosion "
                f"({stage.records_read:,} read -> {stage.records_written:,} written)."
            )
        if stage.shuffle_output_bytes_spilled > 0:
            spilled_mb = stage.shuffle_output_bytes_spilled / (1024 * 1024)
            reasons.append(
                f"Stage '{stage.name}' spilled {spilled_mb:.1f} MB to persistent disk."
            )

    # Check SQL text for explicit CROSS JOIN or missing join predicates
    if _CROSS_JOIN_PATTERN.search(query_text):
        reasons.append("Query contains explicit CROSS JOIN syntax.")
    if _TAUTOLOGY_JOIN_PATTERN.search(query_text):
        reasons.append("Query contains unconditional join predicate (e.g. ON 1=1).")

    return (bool(reasons), reasons)


def detect_slot_contention(
    insights: PerformanceInsights,
    stages: List[JobStage],
    total_slot_ms: int,
) -> Tuple[bool, List[str]]:
    """Detects whether a query suffered from slot starvation or queueing.

    Args:
        insights: Performance insights extracted from query_info.
        stages: List of query execution stages.
        total_slot_ms: Total slot compute time in milliseconds.

    Returns:
        Tuple of (is_contention, list_of_reasons).
    """
    reasons: List[str] = []

    if insights.slot_contention:
        reasons.append(
            "BigQuery Performance Insights flagged explicit slot contention."
        )

    for stage in stages:
        if stage.wait_ratio_avg > 0.40 and total_slot_ms > 60_000:
            reasons.append(
                f"Stage '{stage.name}' spent {stage.wait_ratio_avg * 100:.1f}% "
                "of its time waiting for available slots."
            )

    return (bool(reasons), reasons)


def detect_unpartitioned_scans(
    total_bytes_billed: int, query_text: str
) -> Tuple[bool, List[str]]:
    """Detects queries scanning large datasets without partition filters.

    Args:
        total_bytes_billed: Total bytes billed for the query.
        query_text: Raw SQL text of the query.

    Returns:
        Tuple of (is_unpartitioned, list_of_reasons).
    """
    reasons: List[str] = []
    ten_gb = 10 * 1024 * 1024 * 1024

    if total_bytes_billed >= ten_gb:
        scanned_gb = total_bytes_billed / (1024 * 1024 * 1024)
        has_where = bool(_WHERE_CLAUSE_PATTERN.search(query_text))

        if has_where and _FUNCTION_WRAPPER_ANTIPATTERN.search(query_text):
            reasons.append(
                f"Scanned {scanned_gb:.1f} GB with function wrapper on date/timestamp column in WHERE clause, preventing partition pruning."
            )
        elif not has_where or not _VALID_PARTITION_FILTER_PATTERN.search(query_text):
            reasons.append(
                f"Scanned {scanned_gb:.1f} GB without detectable partition filter."
            )

    return (bool(reasons), reasons)


# ==============================================================================
# Query Analysis & Transformation Engine
# ==============================================================================


def parse_job_row(
    row_data: Dict[str, Any],
    ondemand_rate_per_tib: float,
    editions_rate_per_slot_hour: float,
) -> QueryJobMetrics:
    """Transforms raw INFORMATION_SCHEMA row dictionary into QueryJobMetrics.

    Args:
        row_data: Raw dictionary from INFORMATION_SCHEMA or mock data.
        ondemand_rate_per_tib: Dollar rate per TiB billed.
        editions_rate_per_slot_hour: Dollar rate per slot-hour.

    Returns:
        Populated QueryJobMetrics instance.
    """
    job_id = str(row_data.get("job_id") or "")
    project_id = str(row_data.get("project_id") or "")
    user_email = str(row_data.get("user_email") or "unknown")

    # Parse timestamps
    start_raw = row_data.get("start_time")
    end_raw = row_data.get("end_time")

    def _parse_ts(ts_val: Any) -> datetime:
        if isinstance(ts_val, datetime):
            if ts_val.tzinfo is None:
                return ts_val.replace(tzinfo=timezone.utc)
            return ts_val.astimezone(timezone.utc)
        if isinstance(ts_val, str):
            try:
                # Handle ISO8601 strings
                return datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
            except ValueError:
                pass
        return datetime.now(timezone.utc)

    start_time = _parse_ts(start_raw)
    end_time = _parse_ts(end_raw)

    query_text = str(row_data.get("query") or "")
    total_slot_ms = int(row_data.get("total_slot_ms") or 0)
    total_bytes_billed = int(row_data.get("total_bytes_billed") or 0)
    cache_hit = bool(row_data.get("cache_hit", False))

    # Parse stages
    stages: List[JobStage] = [
        JobStage(
            stage_id=int(s.get("stage_id") or 0),
            name=str(s.get("name") or ""),
            records_read=int(s.get("records_read") or 0),
            records_written=int(s.get("records_written") or 0),
            shuffle_output_bytes=int(s.get("shuffle_output_bytes") or 0),
            shuffle_output_bytes_spilled=int(
                s.get("shuffle_output_bytes_spilled") or 0
            ),
            wait_ratio_avg=float(s.get("wait_ratio_avg") or 0.0),
            slot_ms=int(s.get("slot_ms") or 0),
        )
        for s in (row_data.get("job_stages") or [])
    ]

    # Parse performance insights
    insights = PerformanceInsights()
    raw_insights = row_data.get("performance_insights")
    if isinstance(raw_insights, dict):
        standalone = raw_insights.get("stage_performance_standalone_insights") or []
        for st_insight in standalone:
            if st_insight.get("slot_contention"):
                insights.slot_contention = True
            if st_insight.get("insufficient_shuffle_quota"):
                insights.insufficient_shuffle_quota = True

    # Compute calculations
    slot_hours = calculate_slot_hours(total_slot_ms)
    avg_concurrency = calculate_avg_slot_concurrency(
        total_slot_ms, start_time, end_time
    )
    cost_ondemand, cost_editions = calculate_cost_estimates(
        total_bytes_billed,
        slot_hours,
        ondemand_rate_per_tib=ondemand_rate_per_tib,
        editions_rate_per_slot_hour=editions_rate_per_slot_hour,
    )

    # Apply heuristic detections
    bottlenecks: List[str] = []
    recommendations: List[Dict[str, str]] = []

    # 1. Cartesian Join check
    is_cartesian, cartesian_reasons = detect_cartesian_joins(stages, query_text)
    if is_cartesian:
        bottlenecks.extend(cartesian_reasons)
        recommendations.append(
            {
                "job_id": job_id,
                "rule_id": "JOIN-001",
                "severity": "CRITICAL",
                "title": "Cartesian or Exploding Join Detected",
                "action": (
                    "Eliminate CROSS JOIN or qualify join predicates. "
                    "Pre-aggregate dimension tables before joining to avoid row multiplication."
                ),
            }
        )

    # 2. Slot Contention check
    is_contention, contention_reasons = detect_slot_contention(
        insights, stages, total_slot_ms
    )
    if is_contention:
        bottlenecks.extend(contention_reasons)
        recommendations.append(
            {
                "job_id": job_id,
                "rule_id": "SLOT-001",
                "severity": "HIGH",
                "title": "Severe Slot Contention and Queueing",
                "action": (
                    "Allocate dedicated capacity reservation or enable slot autoscaling. "
                    "Optimize heavy stages to reduce peak concurrent slot requirements."
                ),
            }
        )

    # 3. Unpartitioned Scan check
    is_unpartitioned, unpart_reasons = detect_unpartitioned_scans(
        total_bytes_billed, query_text
    )
    if is_unpartitioned:
        bottlenecks.extend(unpart_reasons)
        recommendations.append(
            {
                "job_id": job_id,
                "rule_id": "PART-001",
                "severity": "HIGH",
                "title": "Expensive Unpartitioned Table Scan",
                "action": (
                    "Partition table by ingestion time or date/timestamp column. "
                    "Enforce partition filters via require_partition_filter = TRUE."
                ),
            }
        )

    return QueryJobMetrics(
        job_id=job_id,
        project_id=project_id,
        user_email=user_email,
        start_time=start_time,
        end_time=end_time,
        query_text=query_text,
        total_slot_ms=total_slot_ms,
        total_bytes_billed=total_bytes_billed,
        cache_hit=cache_hit,
        slot_hours=slot_hours,
        avg_slot_concurrency=avg_concurrency,
        estimated_cost_usd_ondemand=cost_ondemand,
        estimated_cost_usd_editions=cost_editions,
        stages=stages,
        insights=insights,
        bottlenecks=bottlenecks,
        recommendations=recommendations,
    )


def summarize_analysis(
    jobs: List[QueryJobMetrics],
    project_id: str,
    region: str,
    lookback_days: int,
    threshold_slot_hours: float = 0.5,
    mode: str = "all",
) -> AnalysisSummary:
    """Compiles aggregate summary and ranks top resource consumers.

    Args:
        jobs: List of analyzed query metrics.
        project_id: Target Google Cloud project.
        region: Regional qualifier.
        lookback_days: Evaluation window in days.
        threshold_slot_hours: Slot-hours threshold for flagging heavy queries.
        mode: Focus area ('slots', 'cost', 'bottlenecks', 'all').

    Returns:
        Compiled AnalysisSummary instance.
    """
    total_slot_hours = round(sum(j.slot_hours for j in jobs), 4)
    total_cost_ondemand = round(sum(j.estimated_cost_usd_ondemand for j in jobs), 4)
    total_cost_editions = round(sum(j.estimated_cost_usd_editions for j in jobs), 4)

    contention_count = 0
    cartesian_count = 0
    unpartitioned_count = 0
    all_recommendations: List[Dict[str, str]] = []

    for j in jobs:
        for r in j.recommendations:
            all_recommendations.append(r)
            if r.get("rule_id") == "SLOT-001":
                contention_count += 1
            elif r.get("rule_id") == "JOIN-001":
                cartesian_count += 1
            elif r.get("rule_id") == "PART-001":
                unpartitioned_count += 1

    # Filter and rank top heavy jobs according to active analysis mode
    if mode == "cost":
        heavy_jobs = sorted(
            jobs,
            key=lambda x: max(x.estimated_cost_usd_ondemand, x.estimated_cost_usd_editions),
            reverse=True,
        )
    elif mode == "bottlenecks":
        bottlenecked = [j for j in jobs if j.bottlenecks]
        heavy_jobs = sorted(
            bottlenecked if bottlenecked else jobs,
            key=lambda x: (len(x.bottlenecks), x.slot_hours),
            reverse=True,
        )
    else:
        # 'slots' or 'all'
        heavy_jobs = [j for j in jobs if j.slot_hours >= threshold_slot_hours]
        if not heavy_jobs:
            heavy_jobs = list(jobs)
        heavy_jobs.sort(key=lambda x: x.slot_hours, reverse=True)

    return AnalysisSummary(
        project_id=project_id,
        region=region,
        lookback_days=lookback_days,
        total_jobs_analyzed=len(jobs),
        total_slot_hours_consumed=total_slot_hours,
        total_cost_usd_ondemand=total_cost_ondemand,
        total_cost_usd_editions=total_cost_editions,
        contention_job_count=contention_count,
        cartesian_job_count=cartesian_count,
        unpartitioned_job_count=unpartitioned_count,
        top_heavy_jobs=heavy_jobs,
        all_recommendations=all_recommendations,
    )


# ==============================================================================
# SQL Query Generator
# ==============================================================================


def generate_information_schema_query(
    project_id: str, region: str, days: int, limit: int = 50
) -> str:
    """Builds parameterized regional INFORMATION_SCHEMA query string.

    Args:
        project_id: Google Cloud project ID.
        region: Regional qualifier (e.g. 'region-us').
        days: Lookback interval in days.
        limit: Query row limit.

    Returns:
        Formatted GoogleSQL query string.
    """
    normalized = normalize_region(region)
    return f"""SELECT
  project_id,
  job_id,
  user_email,
  start_time,
  end_time,
  query,
  total_slot_ms,
  total_bytes_processed,
  total_bytes_billed,
  cache_hit,
  referenced_tables,
  job_stages,
  query_info.performance_insights AS performance_insights
FROM
  `{project_id}`.`{normalized}`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE
  creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
  AND job_type = 'QUERY'
  AND state = 'DONE'
  AND statement_type != 'SCRIPT'
ORDER BY
  total_slot_ms DESC
LIMIT {limit}"""


# ==============================================================================
# Output Formatters (Table, JSON, CSV)
# ==============================================================================


def format_as_ascii_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Renders a clean ASCII table with column alignment without external dependencies.

    Args:
        headers: List of column header names.
        rows: Matrix of row values.

    Returns:
        Formatted table string.
    """
    str_rows = [[str(val) for val in row] for row in rows]
    col_widths = [len(h) for h in headers]

    for row in str_rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(val))

    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    separator_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    row_lines = [
        " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row))
        for row in str_rows
    ]

    return f"{header_line}\n{separator_line}\n" + "\n".join(row_lines)


def render_table_output(
    summary: AnalysisSummary, limit: int = 10, mode: str = "all"
) -> str:
    """Renders comprehensive terminal table report.

    Args:
        summary: AnalysisSummary object.
        limit: Maximum number of rows to display.
        mode: Focus area ('slots', 'cost', 'bottlenecks', 'all').

    Returns:
        Formatted text report.
    """
    buf = io.StringIO()
    buf.write("==============================================================================\n")
    buf.write("                  BIGQUERY SLOT & COST OPTIMIZER REPORT                       \n")
    buf.write("==============================================================================\n")
    buf.write(
        f"Project: {summary.project_id} | Region: {summary.region} | "
        f"Window: Last {summary.lookback_days} days | Focus Mode: {mode.upper()}\n"
    )
    buf.write(f"Queries Analyzed: {summary.total_jobs_analyzed} | Total Slot-Hours: {summary.total_slot_hours_consumed:.2f}\n")
    buf.write(f"Estimated Cost (On-Demand): ${summary.total_cost_usd_ondemand:.2f} | (Editions): ${summary.total_cost_usd_editions:.2f}\n")
    buf.write(f"Bottlenecks Detected: Contention: {summary.contention_job_count} | Cartesian: {summary.cartesian_job_count} | Unpartitioned: {summary.unpartitioned_job_count}\n")
    buf.write("------------------------------------------------------------------------------\n\n")

    if mode == "cost":
        buf.write("TOP COST-INTENSIVE QUERIES:\n")
    elif mode == "bottlenecks":
        buf.write("TOP BOTTLENECKED QUERIES:\n")
    else:
        buf.write("TOP COMPUTE-INTENSIVE QUERIES:\n")
    headers = ["Job ID", "Slot-Hours", "Avg Slots", "Cost (OD)", "Cost (Ed)", "Bottlenecks"]
    rows = []
    for job in summary.top_heavy_jobs[:limit]:
        b_summary = ", ".join(j["rule_id"] for j in job.recommendations) if job.recommendations else "NONE"
        rows.append([
            job.job_id,
            f"{job.slot_hours:.2f}",
            f"{job.avg_slot_concurrency:.1f}",
            f"${job.estimated_cost_usd_ondemand:.2f}",
            f"${job.estimated_cost_usd_editions:.2f}",
            b_summary,
        ])

    if rows:
        buf.write(format_as_ascii_table(headers, rows))
    else:
        buf.write("No query jobs exceeded the slot threshold.")
    buf.write("\n\n")

    if summary.all_recommendations:
        buf.write("ACTIONABLE RECOMMENDATIONS:\n")
        rec_headers = ["Job ID", "Rule ID", "Severity", "Action"]
        rec_rows = [
            [r.get("job_id", ""), r.get("rule_id", ""), r.get("severity", ""), r.get("action", "")]
            for r in summary.all_recommendations[:limit]
        ]
        buf.write(format_as_ascii_table(rec_headers, rec_rows))
        buf.write("\n")

    buf.write("DISCLAIMER: Cost estimates are indicative baselines and vary by region, edition, and commitments.\n")
    buf.write("For official pricing details, see: https://cloud.google.com/bigquery/pricing\n")

    return buf.getvalue()


def render_json_output(summary: AnalysisSummary) -> str:
    """Renders structured JSON report matching the M1 contract schema.

    Args:
        summary: AnalysisSummary object.

    Returns:
        JSON string.
    """
    total_cost_est = summary.total_cost_usd_ondemand
    payload = {
        "summary": {
            "project_id": summary.project_id,
            "region": summary.region,
            "lookback_days": summary.lookback_days,
            "total_queries_analyzed": summary.total_jobs_analyzed,
            "total_slot_hours": summary.total_slot_hours_consumed,
            "total_cost_usd_est": total_cost_est,
            "total_cost_usd_ondemand": summary.total_cost_usd_ondemand,
            "total_cost_usd_editions": summary.total_cost_usd_editions,
            "contention_job_count": summary.contention_job_count,
            "cartesian_job_count": summary.cartesian_job_count,
            "unpartitioned_job_count": summary.unpartitioned_job_count,
            "pricing_disclaimer": (
                "Cost estimates are indicative baselines and vary by region, edition, "
                "and commitments. See https://cloud.google.com/bigquery/pricing"
            ),
        },
        "pricing_disclaimer": (
            "Cost estimates are indicative baselines and vary by region, edition, "
            "and commitments. See https://cloud.google.com/bigquery/pricing"
        ),
        "recommendations": summary.all_recommendations,
        "top_heavy_jobs": [
            {
                "job_id": j.job_id,
                "project_id": j.project_id,
                "user_email": j.user_email,
                "slot_hours": j.slot_hours,
                "avg_slot_concurrency": j.avg_slot_concurrency,
                "total_bytes_billed": j.total_bytes_billed,
                "estimated_cost_usd_ondemand": j.estimated_cost_usd_ondemand,
                "estimated_cost_usd_editions": j.estimated_cost_usd_editions,
                "bottlenecks": j.bottlenecks,
            }
            for j in summary.top_heavy_jobs
        ],
    }
    return json.dumps(payload, indent=2)


def render_csv_output(summary: AnalysisSummary) -> str:
    """Renders tabular results as comma-separated values.

    Args:
        summary: AnalysisSummary object.

    Returns:
        CSV formatted string.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "job_id",
        "project_id",
        "user_email",
        "slot_hours",
        "avg_slot_concurrency",
        "total_bytes_billed",
        "cost_usd_ondemand",
        "cost_usd_editions",
        "bottlenecks",
    ])
    for j in summary.top_heavy_jobs:
        writer.writerow([
            j.job_id,
            j.project_id,
            j.user_email,
            j.slot_hours,
            j.avg_slot_concurrency,
            j.total_bytes_billed,
            j.estimated_cost_usd_ondemand,
            j.estimated_cost_usd_editions,
            "; ".join(j.bottlenecks),
        ])
    return output.getvalue()


# ==============================================================================
# Execution & Ingestion Dispatchers
# ==============================================================================


def load_mock_data(mock_path: str) -> List[Dict[str, Any]]:
    """Loads and validates synthetic offline telemetry JSON file.

    Args:
        mock_path: Path to synthetic telemetry file.

    Returns:
        List of raw job dictionaries.

    Raises:
        FileNotFoundError: If mock_path does not exist.
        ValueError: If mock data is not a valid list or invalid JSON.
    """
    if not os.path.exists(mock_path):
        raise FileNotFoundError(f"Mock data file not found: {mock_path}")

    with open(mock_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Mock data must contain a list of job dicts, got: {type(data)}")

    return data


def fetch_bigquery_jobs(
    project_id: str, region: str, days: int, limit: int = 50
) -> List[Dict[str, Any]]:
    """Executes live query against INFORMATION_SCHEMA via BigQuery client.

    Args:
        project_id: Google Cloud project ID.
        region: Regional qualifier.
        days: Lookback interval in days.
        limit: Query row limit.

    Returns:
        List of row dictionaries.
    """
    try:
        from google.cloud import bigquery
        from google.auth.exceptions import DefaultCredentialsError
        from google.api_core.exceptions import Forbidden
    except ImportError as e:
        raise ImportError(f"google-cloud-bigquery library required for live execution: {e}")

    try:
        client = bigquery.Client(project=project_id)
        sql = generate_information_schema_query(project_id, region, days, limit)
        query_job = client.query(sql)
        results = query_job.result()
        return [dict(row) for row in results]
    except DefaultCredentialsError as e:
        print("ERROR: Authentication failed. No Application Default Credentials found.", file=sys.stderr)
        print("Remediation: Run 'gcloud auth application-default login' to authenticate.", file=sys.stderr)
        sys.exit(2)
    except Forbidden as e:
        print(f"ERROR: Permission denied querying INFORMATION_SCHEMA for project '{project_id}': {e}", file=sys.stderr)
        print("Remediation: Ensure caller has 'roles/bigquery.resourceViewer' or 'roles/bigquery.user'.", file=sys.stderr)
        sys.exit(3)


# ==============================================================================
# CLI Argument Parser & Entrypoint
# ==============================================================================


def create_argument_parser() -> argparse.ArgumentParser:
    """Builds command-line argument specification."""
    parser = argparse.ArgumentParser(
        prog="slot_analyzer.py",
        description="BigQuery Slot and Cost Optimizer CLI.",
    )
    parser.add_argument(
        "--project",
        "--project-id",
        dest="project_id",
        type=str,
        default=None,
        help="Target Google Cloud project ID.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="region-us",
        help="Regional qualifier for INFORMATION_SCHEMA (default: region-us).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Historical lookback window in days (1 to 30, default: 7).",
    )
    parser.add_argument(
        "--mode",
        choices=["slots", "cost", "bottlenecks", "all"],
        default="all",
        help="Focus area of analysis (default: all).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of queries to display (default: 10).",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output rendering format (default: table).",
    )
    parser.add_argument(
        "--threshold-slot-hours",
        type=float,
        default=0.5,
        help="Minimum slot-hours threshold to flag query as heavy (default: 0.5).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print SQL template without executing against BigQuery.",
    )
    parser.add_argument(
        "--mock-data-file",
        type=str,
        default=None,
        help="Path to synthetic offline JSON file for testing without GCP credentials.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to write formatted output.",
    )
    parser.add_argument(
        "--ondemand-rate",
        type=float,
        default=None,
        help=(
            "On-demand pricing rate in USD per TiB (required unless --dry-run). "
            "Agents must retrieve live regional rates at runtime from "
            "https://cloud.google.com/bigquery/pricing and pass explicitly."
        ),
    )
    parser.add_argument(
        "--slot-hour-rate",
        type=float,
        default=None,
        help=(
            "Editions pricing rate in USD per slot-hour (required unless --dry-run). "
            "Agents must retrieve live regional rates at runtime from "
            "https://cloud.google.com/bigquery/pricing and pass explicitly."
        ),
    )
    return parser


def run_analysis(args: argparse.Namespace) -> int:
    """Orchestrates argument validation, data ingestion, analysis, and rendering.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit code (0: success, 1: execution error, 2: validation error, 3: auth/permission).
    """
    # 1. Validate argument boundaries
    if args.days < 1 or args.days > 30:
        print(f"Validation Error: --days must be between 1 and 30, got: {args.days}", file=sys.stderr)
        return 2

    region = normalize_region(args.region)
    project_id = args.project_id or os.getenv("GOOGLE_CLOUD_PROJECT")

    # 2. Dry-Run inspection mode
    if args.dry_run:
        if not project_id:
            print(
                "Validation Error: --project-id (or GOOGLE_CLOUD_PROJECT environment variable) is required.",
                file=sys.stderr,
            )
            return 2
        sql = generate_information_schema_query(project_id, region, args.days, args.limit * 5)
        print("-- DRY-RUN: Regional BigQuery INFORMATION_SCHEMA Query:")
        print(sql)
        return 0

    if args.ondemand_rate is None or args.slot_hour_rate is None:
        print(
            "Validation Error: --ondemand-rate and --slot-hour-rate are required. "
            "Retrieve live regional pricing rates from "
            "https://cloud.google.com/bigquery/pricing and pass them explicitly.",
            file=sys.stderr,
        )
        return 2

    # 3. Data acquisition (offline mock vs live BigQuery)
    try:
        if args.mock_data_file:
            raw_rows = load_mock_data(args.mock_data_file)
            if not project_id:
                project_id = (
                    str(raw_rows[0]["project_id"])
                    if raw_rows and "project_id" in raw_rows[0]
                    else "mock-project"
                )
        else:
            if not project_id:
                print(
                    "Validation Error: --project-id (or GOOGLE_CLOUD_PROJECT environment variable) is required.",
                    file=sys.stderr,
                )
                return 2
            raw_rows = fetch_bigquery_jobs(project_id, region, args.days, args.limit * 5)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Execution Error: {e}", file=sys.stderr)
        return 1

    # 4. Processing & Metrics Compilation
    ondemand_rate = float(args.ondemand_rate)
    slot_hour_rate = float(args.slot_hour_rate)
    analyzed_jobs = [
        parse_job_row(
            row,
            ondemand_rate_per_tib=ondemand_rate,
            editions_rate_per_slot_hour=slot_hour_rate,
        )
        for row in raw_rows
    ]
    analysis_mode = getattr(args, "mode", "all")
    summary = summarize_analysis(
        jobs=analyzed_jobs,
        project_id=project_id,
        region=region,
        lookback_days=args.days,
        threshold_slot_hours=args.threshold_slot_hours,
        mode=analysis_mode,
    )

    # 5. Output rendering
    if args.format == "json":
        rendered = render_json_output(summary)
    elif args.format == "csv":
        rendered = render_csv_output(summary)
    else:
        rendered = render_table_output(summary, limit=args.limit, mode=analysis_mode)

    if args.output_file:
        try:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(rendered)
        except OSError as e:
            print(f"File Error: Failed to write to {args.output_file}: {e}", file=sys.stderr)
            return 1
    else:
        print(rendered)

    return 0


def main() -> None:
    """CLI script entrypoint."""
    parser = create_argument_parser()
    args = parser.parse_args()
    exit_code = run_analysis(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
