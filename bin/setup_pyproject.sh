#!/usr/bin/env bash
# bin/setup_pyproject.sh
#
# Unified interactive bootstrap for vailá: detects OS/arch/NVIDIA, picks the
# PyTorch dependency group (`cpu` or `cuda`) and the optional extras, then runs
# `uv sync`.
#
# There is a SINGLE committed pyproject.toml / uv.lock for every machine and OS:
# the hardware choice is a uv dependency group, not a file swap. Nothing in the
# repository is modified by this script, so there is never a machine-specific
# manifest to keep out of git.
#
#   cpu   -> torch/torchvision/torchaudio from https://download.pytorch.org/whl/cpu
#            (on macOS: the PyPI wheel, which is the Metal/MPS build)
#   cuda  -> the same trio from https://download.pytorch.org/whl/cu128 + tensorrt
#            + nvidia-ml-py (NVIDIA only; not available on macOS)
#
# Cross-platform (bash): Linux, macOS, WSL, Git Bash / MSYS2 on Windows.
# For native Windows PowerShell, use bin/setup_pyproject.ps1.
#
# Usage:
#   bin/setup_pyproject.sh                                # interactive, auto-detect
#   bin/setup_pyproject.sh --target=cuda --extras=sam
#   bin/setup_pyproject.sh --target=cpu --non-interactive --yes
#   bin/setup_pyproject.sh --help
#
# Flags:
#   --target=auto|cpu|cuda     (default: auto; legacy names linux-cuda, win-cuda,
#                              macos and gpu are still accepted)
#   --extras=a,b,c     Comma-separated extras (sam, fifa, sapiens, upscaler, dev)
#   --non-interactive  Do not prompt; use detected/given values
#   --yes, -y          Accept all suggested defaults (interactive but no prompts)
#   --lock             Re-run `uv lock` before syncing (normally unnecessary)
#   --no-lock          Skip `uv lock` (default)
#   --no-sync          Skip `uv sync`
#   --help, -h         Show this help and exit

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---------- defaults ----------
TARGET="auto"
EXTRAS_CLI=""
NON_INTERACTIVE=0
ACCEPT_DEFAULTS=0
RUN_LOCK=0
RUN_SYNC=1

# ---------- args ----------
print_help() {
    sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'
}

for arg in "$@"; do
    case "$arg" in
        --target=*)        TARGET="${arg#*=}" ;;
        --extras=*)        EXTRAS_CLI="${arg#*=}" ;;
        --full|--preset=full) EXTRAS_CLI="all" ;;
        --non-interactive) NON_INTERACTIVE=1 ;;
        -y|--yes)          ACCEPT_DEFAULTS=1 ;;
        --lock)            RUN_LOCK=1 ;;
        --no-lock)         RUN_LOCK=0 ;;
        --no-sync)         RUN_SYNC=0 ;;
        --skip-worktree|--no-skip-worktree)
            # Kept for backward compatibility: pyproject.toml / uv.lock are now
            # identical on every machine, so there is nothing to hide from git.
            warn "note: $arg is obsolete (single portable pyproject.toml/uv.lock); ignoring."
            ;;
        -h|--help)         print_help; exit 0 ;;
        *)
            echo "error: unknown argument: $arg" >&2
            print_help >&2
            exit 2
            ;;
    esac
done

# ---------- colors (optional, ignored if not a TTY) ----------
if [[ -t 1 ]] && [[ "${NO_COLOR:-}" == "" ]]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
    CYAN=$'\033[36m'; RED=$'\033[31m'; RESET=$'\033[0m'
else
    BOLD=""; DIM=""; GREEN=""; YELLOW=""; CYAN=""; RED=""; RESET=""
fi

say()  { printf '%s\n' "$*"; }
info() { printf '%s%s%s\n' "$CYAN"   "$*" "$RESET"; }
ok()   { printf '%s%s%s\n' "$GREEN"  "$*" "$RESET"; }
warn() { printf '%s%s%s\n' "$YELLOW" "$*" "$RESET" >&2; }
err()  { printf '%s%s%s\n' "$RED"    "$*" "$RESET" >&2; }

# ---------- detection ----------
detect_os() {
    case "$(uname -s 2>/dev/null || echo unknown)" in
        Linux)
            if grep -qiE '(microsoft|wsl)' /proc/version 2>/dev/null; then
                echo "wsl"
            else
                echo "linux"
            fi
            ;;
        Darwin)                       echo "macos" ;;
        MINGW*|MSYS*|CYGWIN*)         echo "windows" ;;
        *)
            if [[ "${OS:-}" == "Windows_NT" ]]; then echo "windows"
            else echo "unknown"
            fi
            ;;
    esac
}

detect_arch() {
    uname -m 2>/dev/null || echo "unknown"
}

detect_nvidia() {
    if command -v nvidia-smi >/dev/null 2>&1 \
        && nvidia-smi -L 2>/dev/null | grep -qi '^GPU '; then
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null \
            | head -n1
        return 0
    fi
    return 1
}

OS="$(detect_os)"
ARCH="$(detect_arch)"
GPU_INFO=""
if GPU_INFO="$(detect_nvidia 2>/dev/null)"; then
    HAS_NVIDIA=1
else
    HAS_NVIDIA=0
    GPU_INFO=""
fi

# ---------- auto target selection ----------
auto_target() {
    case "$OS" in
        macos)            echo "cpu" ;;   # PyPI wheel = Metal/MPS build
        windows|linux|wsl) [[ "$HAS_NVIDIA" == 1 ]] && echo "cuda" || echo "cpu" ;;
        *)                echo "cpu" ;;
    esac
}

if [[ "$TARGET" == "auto" ]]; then
    TARGET="$(auto_target)"
fi

# ---------- target -> torch dependency group ----------
# Legacy target names (linux-cuda / win-cuda / macos / gpu) map onto the two
# groups declared in pyproject.toml so old commands and docs keep working.
normalize_target() {
    case "$1" in
        cpu|macos|mac|metal|mps)          echo "cpu" ;;
        cuda|gpu|linux-cuda|win-cuda|nvidia) echo "cuda" ;;
        *)
            err "unknown target: $1"
            err "valid: cpu, cuda (legacy: linux-cuda, win-cuda, macos)"
            exit 2
            ;;
    esac
}

detect_installed_extras() {
    local detected=""
    local py="$ROOT/.venv/bin/python3"
    if [[ -x "$py" ]]; then
        if "$py" -c "import sam3" 2>/dev/null; then detected+=" sam"; fi
        if "$py" -c "import sapiens" 2>/dev/null; then detected+=" sapiens"; fi
        if "$py" -c "import pytorch_lightning" 2>/dev/null; then detected+=" fifa"; fi
        if "$py" -c "import diffusers" 2>/dev/null; then detected+=" upscaler"; fi
        if "$py" -c "import pytest" 2>/dev/null; then detected+=" dev"; fi
    fi
    echo "$detected"
}

suggested_extras_for_target() {
    # tensorrt / nvidia-ml-py now live in the `cuda` dependency group, not in an
    # extra, so the suggestion is purely "whatever is already installed".
    local base=""
    local installed
    installed="$(detect_installed_extras)"
    local combined="$base $installed"
    # Deduplicate whitespace-separated words
    echo "$combined" | tr ' ' '\n' | awk 'NF && !seen[$0]++' | tr '\n' ' ' | sed 's/ $//'
}

TARGET="$(normalize_target "$TARGET")"
if [[ "$TARGET" == "cuda" && "$OS" == "macos" ]]; then
    err "target 'cuda' is not available on macOS (no CUDA wheels); use --target=cpu."
    exit 2
fi

SUGGESTED_EXTRAS="$(suggested_extras_for_target "$TARGET")"

# ---------- summary ----------
echo ""
echo "${BOLD}== vailá pyproject setup ==${RESET}"
printf '  %-18s %s\n' "OS detected:"   "$OS ($ARCH)"
if [[ -n "$GPU_INFO" ]]; then
    printf '  %-18s %s\n' "NVIDIA GPU:" "$GPU_INFO"
else
    printf '  %-18s %s\n' "NVIDIA GPU:" "${DIM}none detected${RESET}"
fi
printf '  %-18s %s\n' "Target:"           "${BOLD}${TARGET}${RESET}"
printf '  %-18s %s\n' "PyTorch group:"    "$TARGET (uv sync --group $TARGET)"
printf '  %-18s %s\n' "Suggested extras:" "${SUGGESTED_EXTRAS:-${DIM}none${RESET}}"
echo ""

# ---------- target confirmation (interactive) ----------
ask() {
    # $1=prompt $2=default(Y/n style)
    local prompt="$1"; local def="${2:-Y}"; local reply
    if [[ "$NON_INTERACTIVE" == 1 || "$ACCEPT_DEFAULTS" == 1 ]]; then
        [[ "$def" =~ ^[Yy]$ ]] && return 0 || return 1
    fi
    local hint="[Y/n]"; [[ "$def" =~ ^[Nn]$ ]] && hint="[y/N]"
    read -r -p "$prompt $hint " reply || true
    reply="${reply:-$def}"
    [[ "$reply" =~ ^[Yy]$ ]]
}

if [[ "$NON_INTERACTIVE" != 1 && "$ACCEPT_DEFAULTS" != 1 ]]; then
    if ! ask "Use target '$TARGET'?" "Y"; then
        echo "Available targets: cpu (CPU wheels / macOS Metal), cuda (NVIDIA CUDA 12.8)"
        read -r -p "Pick target: " new_target
        TARGET="$(normalize_target "${new_target:-$TARGET}")"
        SUGGESTED_EXTRAS="$(suggested_extras_for_target "$TARGET")"
        info "Switched to target=$TARGET, suggested extras='$SUGGESTED_EXTRAS'"
    fi
fi

# ---------- extras selection ----------
# Everything defined in [project.optional-dependencies]; identical on every
# platform, because there is only one pyproject.toml.
AVAILABLE_EXTRAS="sam fifa sapiens upscaler dev"

if [[ "$EXTRAS_CLI" == "all" ]]; then
    EXTRAS="$AVAILABLE_EXTRAS"
elif [[ -n "$EXTRAS_CLI" ]]; then
    EXTRAS="$(echo "$EXTRAS_CLI" | tr ',' ' ' | tr -s ' ')"
elif [[ "$NON_INTERACTIVE" == 1 || "$ACCEPT_DEFAULTS" == 1 ]]; then
    EXTRAS="$SUGGESTED_EXTRAS"
else
    echo ""
    info "Available extras: $AVAILABLE_EXTRAS"
    echo "  sam      = SAM 3 video segmentation (sam3==0.1.3; CUDA at runtime)"
    echo "  fifa     = FIFA Skeletal Tracking Light (pytorch-lightning, timm, ...)"
    echo "  sapiens  = Sapiens2 Pose (transformers + safetensors; CUDA; then bash bin/setup_sapiens2.sh)"
    echo "  upscaler = diffusers (image upscaling)"
    echo "  dev      = ruff, ty, pytest (developer tooling)"
    read -r -p "Extras to install [default: '$SUGGESTED_EXTRAS']: " user_extras
    EXTRAS="${user_extras:-$SUGGESTED_EXTRAS}"
    EXTRAS="$(echo "$EXTRAS" | tr ',' ' ' | tr -s ' ')"
fi

# Validate extras against the chosen template
VALID_EXTRAS=""
INVALID_EXTRAS=""
for e in $EXTRAS; do
    [[ -z "$e" ]] && continue
    if printf ' %s ' $AVAILABLE_EXTRAS | grep -q " $e "; then
        VALID_EXTRAS+="$e "
    else
        INVALID_EXTRAS+="$e "
    fi
done
if [[ -n "$INVALID_EXTRAS" ]]; then
    warn "Ignoring extras not defined in pyproject.toml: $INVALID_EXTRAS"
fi
EXTRAS="$(echo "$VALID_EXTRAS" | tr -s ' ' | sed 's/^ //;s/ $//')"

# ---------- lock ----------
if [[ "$RUN_LOCK" == 1 ]]; then
    if ! command -v uv >/dev/null 2>&1; then
        err "uv not found in PATH. Install uv: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    info "Running: uv lock"
    uv lock
else
    info "Using the committed uv.lock (pass --lock to re-resolve)"
fi

# ---------- sync ----------
if [[ "$RUN_SYNC" == 1 ]]; then
    # `cpu` is a default group in pyproject.toml, so the CUDA build must both
    # drop it and request `cuda` (the two groups are declared as conflicting).
    GROUP_ARGS=()
    if [[ "$TARGET" == "cuda" ]]; then
        GROUP_ARGS=(--no-group cpu --group cuda)
    fi
    SYNC_CMD=(uv sync "${GROUP_ARGS[@]}")
    for e in $EXTRAS; do
        SYNC_CMD+=(--extra "$e")
    done
    info "Running: ${SYNC_CMD[*]}"
    "${SYNC_CMD[@]}"

    # ---- verify + repair CUDA wheel integrity (linux-cuda / win-cuda only) ----
    # Real bug hit in production: uv sync can report "nothing to do" for an
    # nvidia-*-cu12 package whose dist-info is present but whose actual .so
    # payload is missing on disk (broken hardlink / interrupted extraction /
    # disk full) -- `import torch` then fails with e.g.
    # "ImportError: libcusparseLt.so.0: cannot open shared object file",
    # invisible to uv's own "already satisfied" bookkeeping.
    if [[ "$TARGET" == "cuda" ]]; then
        info "Verifying NVIDIA/PyTorch CUDA wheel integrity..."
        BROKEN="$(uv run python bin/verify_cuda_libs.py --quiet 2>/dev/null || true)"
        if [[ -n "$BROKEN" ]]; then
            warn "Corrupted CUDA wheels detected (metadata present, files missing): $(echo "$BROKEN" | tr '\n' ' ')"
            warn "Reinstalling only the broken packages..."
            REPAIR_CMD=(uv sync "${GROUP_ARGS[@]}")
            for pkg in $BROKEN; do REPAIR_CMD+=(--reinstall-package "$pkg"); done
            for e in $EXTRAS; do REPAIR_CMD+=(--extra "$e"); done
            info "Running: ${REPAIR_CMD[*]}"
            "${REPAIR_CMD[@]}"
            STILL_BROKEN="$(uv run python bin/verify_cuda_libs.py --quiet 2>/dev/null || true)"
            if [[ -n "$STILL_BROKEN" ]]; then
                err "Still broken after reinstall: $(echo "$STILL_BROKEN" | tr '\n' ' ')"
                err "Check disk space (df -h) and, if uv cache + .venv are on different"
                err "filesystems, try: export UV_LINK_MODE=copy   then re-run this script."
            else
                ok "CUDA wheel integrity repaired."
            fi
        else
            ok "CUDA wheel integrity verified."
        fi
        uv run python -c "import torch; print('torch', torch.__version__, '- CUDA available:', torch.cuda.is_available())" \
            || warn "torch import still failing after CUDA wheel repair -- see errors above."
    fi

    # ---- verify + repair the Sapiens2 editable install ----
    # Real bug hit in production: `uv sync` (even with --extra sapiens) does
    # not know about the local editable checkout at .local/third_party/sapiens2
    # -- a plain sync can silently drop it. Re-register it if the checkout
    # exists on disk but the package no longer imports (cheap, no network).
    if printf ' %s ' $EXTRAS | grep -q ' sapiens '; then
        if ! uv run python -c "import sapiens" >/dev/null 2>&1; then
            if [[ -d "$ROOT/.local/third_party/sapiens2" ]]; then
                warn "sapiens checkout exists but is not importable -- re-registering editable install..."
                uv pip install -e "$ROOT/.local/third_party/sapiens2" \
                    && ok "sapiens editable install repaired." \
                    || err "Failed to repair sapiens editable install; run: bash bin/setup_sapiens2.sh"
            else
                info "sapiens extra requested but no checkout found yet; run: bash bin/setup_sapiens2.sh"
            fi
        fi
    fi

    ok ""
    ok "Done. vailá ready for target='$TARGET' with extras=[$EXTRAS]."
    say "Run the GUI:   uv run vaila.py"
else
    info "Skipping 'uv sync' (--no-sync)"
    say ""
    say "Next, run manually:"
    GROUP_HINT=""
    [[ "$TARGET" == "cuda" ]] && GROUP_HINT="--no-group cpu --group cuda "
    if [[ -n "$EXTRAS" ]]; then
        say "  uv sync ${GROUP_HINT}$(echo "$EXTRAS" | sed 's/[^ ][^ ]*/--extra &/g')"
    else
        say "  uv sync ${GROUP_HINT}"
    fi
fi

# ---------- clear any legacy skip-worktree bits ----------
# Older versions of this script hid pyproject.toml / uv.lock from git status.
# Both files are portable now, so make sure they are tracked normally again.
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git update-index --no-skip-worktree pyproject.toml uv.lock 2>/dev/null || true
fi
