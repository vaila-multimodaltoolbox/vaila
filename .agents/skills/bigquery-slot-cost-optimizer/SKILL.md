---
name: bigquery-slot-cost-optimizer
description: >-
  Analyzes Google Cloud BigQuery slot consumption, query costs, and execution
  bottlenecks using INFORMATION_SCHEMA. Use when diagnosing slow BigQuery queries,
  slot starvation, high on-demand query costs, unpartitioned table scans, or join
  performance issues. Don't use for generic BigQuery administration (use
  bigquery-basics), BigQuery ML (use bigquery-ai-ml), or DataFrame operations
  (use bigquery-bigframes).
metadata:
  version: 1.0.0
  publisher: google
  category: BigDataAndAnalytics
  tags:
    - bigquery
    - performance
    - cost-optimization
    - slot-analysis
    - sql
---

# BigQuery slot and cost optimizer

This skill equips AI agents and cloud engineers with procedural heuristics to analyze BigQuery resource consumption, calculate slot hours, identify slot contention and queueing, mitigate Cartesian joins, and optimize unpartitioned table scans.

## Trigger conditions and intent mapping

Activate this skill whenever the user asks to:
- "Optimize BigQuery query performance or reduce slot usage"
- "Find the most expensive queries in BigQuery"
- "Diagnose BigQuery slot contention or queueing"
- "Fix slow running BigQuery jobs or memory spillage"
- "Detect Cartesian joins or row count explosions in BigQuery"
- "Identify unpartitioned table scans or missing partition filters"

## Prerequisites and environment setup

Before executing this skill, ensure the environment is configured with the necessary SDKs, permissions, and billing:

1. **Cloud SDK and client library installation**:
   - Install the Google Cloud CLI: [Google Cloud SDK installation guide](https://docs.cloud.google.com/sdk/docs/install-sdk.md.txt)
   - Install the BigQuery Python client:

     ```bash
     pip install google-cloud-bigquery
     ```

1. **Project, billing, and regional selection**:
   - Set the active project:

     ```bash
     gcloud config set project <PROJECT_ID>
     ```

   - **Important**: the target Google Cloud project must have an active Cloud Billing account attached.
   - **Regional selection**: specify the target BigQuery dataset location or execution region, as BigQuery `INFORMATION_SCHEMA` views are strictly region-scoped (for example, multi-regions like `region-us` or `region-eu`, or single regions like `region-us-central1`). Querying the wrong region returns empty job telemetry. Pass the matching region via `--region` (the script automatically normalizes location names like `us-central1` to `region-us-central1`). For valid location identifiers, see [BigQuery locations](https://docs.cloud.google.com/bigquery/docs/locations.md.txt).

1. **API enablement**:
   - Enable the BigQuery API on the project:

     ```bash
     gcloud services enable bigquery.googleapis.com
     ```

1. **Authentication setup**:
   - Authenticate the local gcloud environment and configure Application Default Credentials (ADC):

     ```bash
     gcloud auth login
     gcloud auth application-default login
     ```

1. **IAM roles and permissions**:
   - The executing principal requires the following minimum IAM roles:
     - `roles/bigquery.jobUser`: grants permission to run queries and analyze telemetry.
     - `roles/bigquery.resourceViewer`: grants read-only access to query metadata in `INFORMATION_SCHEMA.JOBS_BY_PROJECT` and capacity reservations.

1. **Pricing reference**:
   - Cost estimates in this skill are for planning purposes. Before running `scripts/slot_analyzer.py`, retrieve live BigQuery billing rates at runtime from official [Google Cloud BigQuery Pricing](https://cloud.google.com/bigquery/pricing) (and consult [BigQuery editions introduction](https://docs.cloud.google.com/bigquery/docs/editions-intro.md.txt) for edition capabilities) after considering user-specific parameters such as target region, chosen edition (`Standard`, `Enterprise`, `Enterprise Plus`), and commitment tier (`Pay-as-you-go`, `1-year`, `3-year`). Pass these runtime-fetched rates explicitly via `--ondemand-rate <USD_PER_TIB>` and `--slot-hour-rate <USD_PER_SLOT_HOUR>`.

## Diagnostic execution workflow

### Execute automated telemetry extraction

Run `scripts/slot_analyzer.py` to pull and analyze historical query telemetry from `INFORMATION_SCHEMA.JOBS_BY_PROJECT`, passing the runtime-retrieved pricing rates for your specific region, edition, and commitment tier:

```bash
# General analysis passing live regional pricing rates fetched from BigQuery pricing
python3 scripts/slot_analyzer.py --project-id <PROJECT_ID> --days 7 \
  --ondemand-rate <USD_PER_TIB> --slot-hour-rate <USD_PER_SLOT_HOUR> --format table

# Output structured JSON for programmatically parsing recommendations
python3 scripts/slot_analyzer.py --project-id <PROJECT_ID> --days 7 \
  --ondemand-rate <USD_PER_TIB> --slot-hour-rate <USD_PER_SLOT_HOUR> --format json

# Offline verification mode using synthetic or extracted telemetry
python3 scripts/slot_analyzer.py --mock-data-file path/to/extracted_telemetry.json \
  --ondemand-rate <USD_PER_TIB> --slot-hour-rate <USD_PER_SLOT_HOUR> --format table

# Dry-run mode to inspect regional SQL query
python3 scripts/slot_analyzer.py --project-id <PROJECT_ID> --region region-us --dry-run
```

Run `python3 scripts/slot_analyzer.py --help` to inspect all supported CLI flags, focus modes (`--mode`), and required pricing rate arguments (`--ondemand-rate` per TiB and `--slot-hour-rate` per slot-hour).

## Metric interpretation and decision tree

Evaluate the telemetry output using the following decision rules. **CRITICAL MANDATE: After classifying the query issue using the decision tree below, you MUST immediately call `view_file` on [references/remediation_playbooks.md](references/remediation_playbooks.md) to read and execute the corresponding remediation playbook (`Rule SLOT-001`, `Rule JOIN-001`, or `Rule PART-001`) and include all mandatory diagnostic SQL queries and 4-step checklists in your response.**

```
[Query Telemetry Analyzed]
       |
       +---> If wait_ratio_avg > 0.40 OR slot_contention == TRUE
       |     --> Classify as slot contention and queueing (Rule SLOT-001)
       |     --> MANDATORY: Read Rule SLOT-001 in references/remediation_playbooks.md
       |
       +---> If shuffle_output_bytes_spilled > 0 OR records_written > 10 * records_read
       |     --> Classify as Cartesian join (Rule JOIN-001)
       |     --> MANDATORY: Read Rule JOIN-001 in references/remediation_playbooks.md
       |
       +---> If total_bytes_billed > 10 GB AND no date/partition filters
       |     --> Classify as unpartitioned scan (Rule PART-001)
       |     --> MANDATORY: Read Rule PART-001 in references/remediation_playbooks.md
       |
       +---> Otherwise
             --> Check BI Engine, search indexes, or materialized view opportunities
             --> MANDATORY: Read references/optimization_rules.md
```

## Remediation playbooks and architectural reference links

To minimize token consumption in `SKILL.md`, concrete remediation playbooks (`Rule SLOT-001`, `Rule JOIN-001`, `Rule PART-001`), diagnostic SQL queries, and DDL rewrite patterns are housed in `references/`:
- **Rule SLOT-001 (Slot contention and queueing)**: [Rule SLOT-001 playbook](references/remediation_playbooks.md#rule-slot-001-slot-contention-and-queueing)
- **Rule JOIN-001 (Cartesian and exploding joins)**: [Rule JOIN-001 playbook](references/remediation_playbooks.md#rule-join-001-cartesian-and-exploding-joins)
- **Rule PART-001 (Unpartitioned scans and partition pruning)**: [Rule PART-001 playbook](references/remediation_playbooks.md#rule-part-001-unpartitioned-scans-and-partition-pruning)
- **Table partitioning, multi-column clustering, and BI Engine**: [optimization rules](references/optimization_rules.md)
- **BigQuery search indexes and materialized views**: [optimization rules](references/optimization_rules.md)

## Verification and validation protocol

Before finalizing query rewrites:

### Dry-run validation

Validate query syntax and calculate estimated bytes scanned without incurring cost:

```python
from google.cloud import bigquery
client = bigquery.Client()
job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
query_job = client.query(optimized_sql, job_config=job_config)
print(f"Scanned bytes: {query_job.total_bytes_processed / (1024**3):.2f} GB")
```

### Offline and dry-run validation

- **Offline mock telemetry verification**: validate heuristic classification, slot contention detection, Cartesian join identification, and cost estimation offline using synthetic or extracted JSON telemetry payloads (`--mock-data-file`):

  ```bash
  python3 scripts/slot_analyzer.py --mock-data-file path/to/extracted_telemetry.json \
    --ondemand-rate <USD_PER_TIB> --slot-hour-rate <USD_PER_SLOT_HOUR> --format table
  ```

- **CLI dry-run inspection**: verify regional SQL query formation and script execution without contacting BigQuery or incurring costs:

  ```bash
  python3 scripts/slot_analyzer.py --project-id <PROJECT_ID> --region region-us --dry-run
  ```
