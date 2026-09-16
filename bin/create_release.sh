#!/usr/bin/env bash
# ==============================================================================
# bin/create_release.sh — Release tagger & multi-OS GitHub Actions launcher
# ==============================================================================
#
# Checks repository state, validates versions, runs test smoke, creates an
# annotated Git tag from main, and pushes it to GitHub to trigger the automated
# VM installer build (.github/workflows/release-installers.yml).
#
# Usage:
#   bash bin/create_release.sh                     # Interactive mode
#   bash bin/create_release.sh --tag=v0.4.3 --yes  # Non-interactive / CI
#   bash bin/create_release.sh --dry-run           # Test validations only
#   bash bin/create_release.sh --help              # Show help
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TAG=""
AUTO_YES=0
DRY_RUN=0
SKIP_TESTS=0

for arg in "$@"; do
    case "$arg" in
        --tag=*)
            TAG="${arg#*=}"
            ;;
        --yes|-y)
            AUTO_YES=1
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        --no-test)
            SKIP_TESTS=1
            ;;
        --help|-h)
            echo "Usage: bash bin/create_release.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --tag=NAME    Git tag name to create (e.g. v0.4.3 or rp15Sep2026)"
            echo "  --yes, -y     Skip confirmation prompt and push tag immediately"
            echo "  --dry-run     Run validations and print plan without creating tag"
            echo "  --no-test     Skip pre-release test smoke check"
            echo "  --help, -h    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            echo "Run 'bash bin/create_release.sh --help' for options." >&2
            exit 1
            ;;
    esac
done

echo "===================================================================="
echo "  vailá — GitHub Release & Multi-OS Installer Launcher"
echo "===================================================================="
echo ""

# 1. Check Git branch
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "❌ Error: Releases must be tagged from the 'main' branch (current: '$CURRENT_BRANCH')." >&2
    echo "   Please checkout main first: git checkout main && git pull origin main" >&2
    exit 1
fi

# 2. Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "❌ Error: Working tree has uncommitted changes." >&2
    echo "   Please commit or stash your changes before tagging a release:" >&2
    git status -s >&2
    exit 1
fi

# 3. Check synchronization between vaila.py and pyproject.toml
VAILA_VERSION="$(grep -E '^Version:' vaila.py | head -1 | awk '{print $2}' | tr -d '\r')"
PYPROJECT_VERSION="$(grep -E '^version *= *"' pyproject.toml | head -1 | sed -E 's/^version *= *"([^"]+)".*/\1/' | tr -d '\r')"

echo "  • Detected vaila.py Version:       $VAILA_VERSION"
echo "  • Detected pyproject.toml Version: $PYPROJECT_VERSION"

if [ -z "$VAILA_VERSION" ] || [ -z "$PYPROJECT_VERSION" ]; then
    echo "❌ Error: Could not resolve package versions." >&2
    exit 1
fi

if [ "$VAILA_VERSION" != "$PYPROJECT_VERSION" ]; then
    echo "⚠️ Warning: vaila.py ($VAILA_VERSION) and pyproject.toml ($PYPROJECT_VERSION) differ!" >&2
    echo "   Recommended: synchronize them before releasing." >&2
fi

# 4. Resolve tag name
DEFAULT_TAG="v${VAILA_VERSION}"
if [ -z "$TAG" ]; then
    if [ "$AUTO_YES" -eq 1 ]; then
        TAG="$DEFAULT_TAG"
    else
        read -r -p "Enter Git tag to create [default: $DEFAULT_TAG]: " USER_TAG
        TAG="${USER_TAG:-$DEFAULT_TAG}"
    fi
fi

# Validate tag pattern for GitHub Actions
if [[ ! "$TAG" =~ ^(v|rp) ]]; then
    echo "⚠️ Warning: Tag '$TAG' does not start with 'v' or 'rp'." >&2
    echo "   Note: .github/workflows/release-installers.yml triggers only on tags matching 'v*' or 'rp*'." >&2
    if [ "$AUTO_YES" -eq 0 ]; then
        read -r -p "Do you wish to proceed anyway? (y/N): " CONT
        if [[ ! "$CONT" =~ ^[yY] ]]; then
            echo "Aborted."
            exit 1
        fi
    fi
fi

# Check if tag already exists locally or remotely
if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
    echo "❌ Error: Tag '$TAG' already exists locally." >&2
    echo "   To overwrite or re-release, delete the tag first: git tag -d $TAG" >&2
    exit 1
fi

# 5. Pre-flight smoke tests
if [ "$SKIP_TESTS" -eq 0 ]; then
    echo ""
    echo "Running pre-flight test smoke check..."
    if ! uv run pytest tests/test_planar_geometry_tracker.py tests/test_video_stabilizer.py tests/test_dlt_rec.py -q; then
        echo "❌ Error: Pre-flight tests failed! Please fix issues before releasing." >&2
        exit 1
    fi
    echo "✅ Tests passed."
fi

echo ""
echo "--------------------------------------------------------------------"
echo "Release Summary:"
echo "  • Branch:       $CURRENT_BRANCH"
echo "  • Tag:          $TAG"
echo "  • Commit:       $(git rev-parse --short HEAD) — $(git log -1 --pretty=%s)"
echo "  • Package Ver:  $VAILA_VERSION"
echo "  • Workflow:     .github/workflows/release-installers.yml"
echo "  • Target VMs:   macOS (vaila_installer.dmg), Windows (vaila_installer.exe)"
echo "--------------------------------------------------------------------"
echo ""

if [ "$DRY_RUN" -eq 1 ]; then
    echo "🔎 Dry-run mode: Tag was NOT created or pushed."
    exit 0
fi

if [ "$AUTO_YES" -eq 0 ]; then
    read -r -p "Create tag '$TAG' and push to origin? (y/N): " CONFIRM
    if [[ ! "$CONFIRM" =~ ^[yY] ]]; then
        echo "Aborted by user."
        exit 0
    fi
fi

echo "Creating annotated Git tag '$TAG'..."
git tag -a "$TAG" -m "Release $TAG (vailá v${VAILA_VERSION})"

echo "Pushing tag '$TAG' to origin..."
git push origin "$TAG"

echo ""
echo "===================================================================="
echo "🎉 Tag '$TAG' successfully pushed!"
echo ""
echo "GitHub Actions VMs are now building the multi-OS installers:"
echo "  👉 Actions:  https://github.com/vaila-multimodaltoolbox/vaila/actions"
echo "  👉 Releases: https://github.com/vaila-multimodaltoolbox/vaila/releases"
echo ""
echo "Once the GitHub Actions workflow finishes (~5-10 min):"
echo "  • vaila_installer.dmg (macOS)"
echo "  • vaila_installer.exe (Windows)"
echo "will be attached automatically to the release."
echo "===================================================================="
