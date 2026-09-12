#!/usr/bin/env bash
# bin/sync_repo.sh
#
# Safe cross-platform git sync for vailá:
# 1. Protects your local uncommitted work (aborts if unrelated files are dirty).
# 2. Temporarily resets local pyproject.toml / uv.lock to the clean CPU version in Git.
# 3. Pulls latest changes from GitHub (git pull --rebase) without merge conflicts.
# 4. Automatically re-applies the right hardware template (Linux CUDA, Windows CUDA, macOS Metal)
#    and runs `uv sync` for your current machine.
#
# Usage:
#   bash bin/sync_repo.sh               # pull + auto-detect hardware + uv sync
#   bash bin/sync_repo.sh --skip-worktree # also hides pyproject.toml from git status
#   bash bin/sync_repo.sh --help
#
set -euo pipefail

for arg in "$@"; do
    case "$arg" in
        -h|--help)
            sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---------- colors ----------
if [[ -t 1 ]] && [[ "${NO_COLOR:-}" == "" ]]; then
    BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
    CYAN=$'\033[36m'; RED=$'\033[31m'; RESET=$'\033[0m'
else
    BOLD=""; GREEN=""; YELLOW=""; CYAN=""; RED=""; RESET=""
fi

info() { printf '%s%s%s\n' "$CYAN"   "$*" "$RESET"; }
ok()   { printf '%s%s%s\n' "$GREEN"  "$*" "$RESET"; }
warn() { printf '%s%s%s\n' "$YELLOW" "$*" "$RESET" >&2; }
err()  { printf '%s%s%s\n' "$RED"    "$*" "$RESET" >&2; }

if ! command -v git >/dev/null 2>&1; then
    err "git is required but not found in PATH."
    exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    err "Not inside a git repository."
    exit 1
fi

# 1. Check for dirty files other than pyproject.toml and uv.lock
git update-index --no-skip-worktree pyproject.toml uv.lock 2>/dev/null || true

DIRTY_FILES="$(git status --porcelain | grep -vE 'pyproject\.toml|uv\.lock' || true)"
if [[ -n "$DIRTY_FILES" ]]; then
    warn "=========================================================================="
    warn "Uncommitted changes detected in repository:"
    echo "$DIRTY_FILES" >&2
    warn ""
    warn "Please commit or stash your code changes before syncing:"
    warn "  git add <files> && git commit -m 'your message'"
    warn "  or: git stash"
    warn "=========================================================================="
    exit 1
fi

# 2. Revert local hardware pyproject.toml/uv.lock to avoid conflicts on pull
info "Resetting local pyproject.toml & uv.lock to clean tracking state..."
git checkout pyproject.toml uv.lock 2>/dev/null || true

# 3. Pull latest changes from remote
info "Pulling latest changes from remote (git pull --rebase)..."
git pull --rebase

# 4. Re-apply hardware setup for this machine
info "Applying hardware configuration for current machine..."
bash "$ROOT/bin/setup_pyproject.sh" "$@"

ok "Repository successfully synchronized and environment is up to date!"
