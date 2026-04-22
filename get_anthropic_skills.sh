#!/usr/bin/env bash
# =============================================================================
# get_anthropic_skills.sh
# Download Anthropic skills directly from GitHub into .claude/skills/
#
# Created by: Paulo Roberto Pereira Santiago
# Date: 17 April 2026
# Version: 1.0.0
# Description: This script downloads Anthropic skills directly from GitHub into .claude/skills/
#
# Usage:
#   chmod +x get_anthropic_skills.sh
#   ./get_anthropic_skills.sh
#
# Description:
# This script downloads Anthropic skills directly from GitHub into .claude/skills/
#
# Run from the ROOT of your vaila project.
# =============================================================================

set -e

BASE_URL="https://raw.githubusercontent.com/anthropics/skills/main/skills"
SKILLS_DIR=".claude/skills"

# Use curl if available, otherwise wget
if command -v curl &>/dev/null; then
    DOWNLOAD="curl -fsSL -o"
elif command -v wget &>/dev/null; then
    DOWNLOAD="wget -q -O"
else
    echo "ERROR: curl or wget is required."
    exit 1
fi

fetch() {
    local url="$1"
    local dest="$2"
    mkdir -p "$(dirname "$dest")"
    echo "  ↓ $dest"
    $DOWNLOAD "$dest" "$url"
}

echo ""
echo "============================================================"
echo "  Downloading Anthropic Skills → $SKILLS_DIR/"
echo "============================================================"
echo ""

# ── 1. mcp-builder ────────────────────────────────────────────────────────────
# Expose vaila's biomechanics analysis functions as MCP tools for AI agents.
# Includes reference files: best practices, Python & Node guides, eval guide.
echo "[ 1/7 ] mcp-builder"
fetch "$BASE_URL/mcp-builder/SKILL.md"                              "$SKILLS_DIR/mcp-builder/SKILL.md"
fetch "$BASE_URL/mcp-builder/reference/mcp_best_practices.md"      "$SKILLS_DIR/mcp-builder/reference/mcp_best_practices.md"
fetch "$BASE_URL/mcp-builder/reference/python_mcp_server.md"       "$SKILLS_DIR/mcp-builder/reference/python_mcp_server.md"
fetch "$BASE_URL/mcp-builder/reference/node_mcp_server.md"         "$SKILLS_DIR/mcp-builder/reference/node_mcp_server.md"
fetch "$BASE_URL/mcp-builder/reference/evaluation.md"              "$SKILLS_DIR/mcp-builder/reference/evaluation.md"

# ── 2. skill-creator ──────────────────────────────────────────────────────────
# Used to create and optimize the vaila-specific custom skills themselves.
echo "[ 2/7 ] skill-creator"
fetch "$BASE_URL/skill-creator/SKILL.md"                           "$SKILLS_DIR/skill-creator/SKILL.md"

# ── 3. xlsx ───────────────────────────────────────────────────────────────────
# Generate rich Excel reports from vaila biomechanics CSV results.
# (Multi-tab workbooks with formulas, charts, formatting)
echo "[ 3/7 ] xlsx"
fetch "$BASE_URL/xlsx/SKILL.md"                                    "$SKILLS_DIR/xlsx/SKILL.md"

# ── 4. pdf ────────────────────────────────────────────────────────────────────
# Generate biomechanics PDF reports combining vaila plots + tables.
echo "[ 4/7 ] pdf"
fetch "$BASE_URL/pdf/SKILL.md"                                     "$SKILLS_DIR/pdf/SKILL.md"

# ── 5. pptx ───────────────────────────────────────────────────────────────────
# Auto-generate PowerPoint presentations of analysis results for researchers.
echo "[ 5/7 ] pptx"
fetch "$BASE_URL/pptx/SKILL.md"                                    "$SKILLS_DIR/pptx/SKILL.md"

# ── 6. webapp-testing ─────────────────────────────────────────────────────────
# Test vaila's HTML documentation and any web dashboards via Playwright.
echo "[ 6/7 ] webapp-testing"
fetch "$BASE_URL/webapp-testing/SKILL.md"                          "$SKILLS_DIR/webapp-testing/SKILL.md"

# ── 7. web-artifacts-builder ──────────────────────────────────────────────────
# Build interactive HTML dashboards for biomechanics data visualization.
echo "[ 7/7 ] web-artifacts-builder"
fetch "$BASE_URL/web-artifacts-builder/SKILL.md"                   "$SKILLS_DIR/web-artifacts-builder/SKILL.md"

echo ""
echo "============================================================"
echo "  ✅  Done! Skills installed in $SKILLS_DIR/"
echo "============================================================"
echo ""
echo "Directory structure:"
find "$SKILLS_DIR" -type f | sort | sed 's|^|  |'
echo ""
echo "Next steps:"
echo "  1. Add $SKILLS_DIR to your .gitignore (or commit it — your choice)"
echo "  2. Reference skills in CLAUDE.md under the 'Skills Available' section"
echo "  3. In any AI assistant, just describe the task and the agent will auto-trigger"
echo "     the right skill (e.g. 'create an Excel report from this CSV')"
echo ""
