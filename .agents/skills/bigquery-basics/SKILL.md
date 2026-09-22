---
name: bigquery-basics
metadata:
  version: "1.0.0"
  category: BigDataAndAnalytics
description: >-
  Manages datasets, tables, and jobs in BigQuery. Use when you need to interact
  with BigQuery, run SQL queries, manage BigQuery resources (datasets, tables,
  views), or perform basic data ingestion and analysis.
---

# BigQuery Basics

BigQuery is a serverless, AI-ready data platform that enables high-speed
analysis of large datasets using SQL and Python. Its disaggregated architecture
separates compute and storage, allowing them to scale independently while
providing built-in machine learning, geospatial analysis, and business
intelligence capabilities.

## Attribution

Tag every BigQuery command you execute directly in the terminal while using
this skill, so usage can be attributed. The tag identifies only the skill and
its version; it carries no user data.

-   Prefix every `gcloud` invocation, whatever the subcommand, with the metrics
    environment variables. Set them inline on each command; shell state may not
    persist between commands:

    ```bash
    CLOUDSDK_METRICS_REQUEST_ATTRIBUTION="agent-skills/1.0 (skill:bigquery-basics)" \
    gcloud <command> [flags]
    ```

    Do not use `gcloud config set` for this: it would persist beyond the current
    task and mislabel unrelated usage.

-   On direct HTTP calls to the BigQuery APIs (for example with `curl`)
    or HTTP requests to the MCP server, append this exact string to the
    end of the existing User-Agent header — the collection pipeline parses
    the `agent-skills/<version>` and `skill:<name>` tokens, so any
    rewording breaks attribution:

    ```
    User-Agent: <existing_user_agent> agent-skills/1.0 (skill:bigquery-basics)
    ```

-   **Constraint for Generated Code**: The tracking prefix is strictly for
    commands you execute directly in the terminal. Do NOT include this tracking
    prefix in generated scripts, Terraform templates, or code provided to the
    user for their own repositories.

## Setup and Basic Usage

1.  **Enable the BigQuery API:**

    ```bash
    CLOUDSDK_METRICS_REQUEST_ATTRIBUTION="agent-skills/1.0 (skill:bigquery-basics)" \
    gcloud services enable bigquery.googleapis.com --quiet
    ```

2.  **Create a Dataset:**

    ```bash
    bq mk --dataset --location=US my_dataset
    ```

3.  **Create a Table:**

    Create a file named `schema.json` with your table schema:

    ```json
    [
      {
        "name": "name",
        "type": "STRING",
        "mode": "REQUIRED"
      },
      {
        "name": "post_abbr",
        "type": "STRING",
        "mode": "NULLABLE"
      }
    ]
    ```

    Then create the table with the `bq` tool:

    ```bash
    bq mk --table my_dataset.mytable schema.json
    ```

4.  **Run a Query:**

    ```bash
    bq query --use_legacy_sql=false \
    'SELECT name FROM `bigquery-public-data.usa_names.usa_1910_2013` \
    WHERE state = "TX" LIMIT 10'
    ```

## Reference Directory

- [Core Concepts](references/core-concepts.md): Storage types, analytics
  workflows, and BigQuery Studio features.

- [Change History](references/change-history.md): Tracking and querying
  incremental table changes using APPENDS and CHANGES.

-   [Continuous Queries](references/continuous-queries.md): Running continuous
    SQL statements to analyze incoming data in real time.

- [CLI Usage](references/cli-usage.md): Essential `bq` command-line tool
  operations for managing data and jobs.

- [Client Libraries](references/client-library-usage.md): Using Google Cloud
  client libraries for Python, Java, Node.js, and Go.

- [MCP Usage](references/mcp-usage.md): Using the BigQuery remote MCP server and
  Gemini CLI extension.

- [Infrastructure as Code](references/iac-usage.md): Terraform examples for
  datasets, tables, and reservations.

- [IAM & Security](references/iam-security.md): Roles, permissions, and data
  governance best practices.

*If you need product information not found in these references, use the
Developer Knowledge MCP server `search_documents` tool.*

## Related Skills

- [BigQuery AI & ML Skill](../bigquery-ai-ml):
  SKILL.md file for BigQuery AI and ML capabilities (forecast, anomaly
  detection, text generation).
