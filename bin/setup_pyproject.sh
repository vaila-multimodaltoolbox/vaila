#!/usr/bin/env bash
# bin/setup_pyproject.sh
#
# Unified interactive bootstrap for vailá: detects OS/arch/NVIDIA, picks the
# right pyproject_*.toml template, lets the user confirm extras, then runs
# `uv lock` + `uv sync --extra ...`.
#
# Cross-platform (bash): Linux, macOS, WSL, Git Bash / MSYS2 on Windows.
# For native Windows PowerShell, use bin/setup_pyproject.ps1.
#
# Usage:
#   bin/setup_pyproject.sh                                # interactive, auto-detect
#   bin/setup_pyproject.sh --target=linux-cuda --extras=gpu,sam
#   bin/setup_pyproject.sh --target=cpu --non-interactive --yes
#   bin/setup_pyproject.sh --help
#
# Flags:
#   --target=auto|cpu|linux-cuda|win-cuda|macos   (default: auto)
#   --extras=a,b,c     Comma-separated extras (gpu, sam, fifa, upscaler, dev)
#   --non-interactive  Do not prompt; use detected/given values
#   --yes, -y          Accept all suggested defaults (interactive but no prompts)
#   --no-lock          Skip `uv lock`
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
RUN_LOCK=1
RUN_SYNC=1

# ---------- args ----------
print_help() {
    sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
}

for arg in "$@"; do
    case "$arg" in
        --target=*)        TARGET="${arg#*=}" ;;
        --extras=*)        EXTRAS_CLI="${arg#*=}" ;;
        --non-interactive) NON_INTERACTIVE=1 ;;
        -y|--yes)          ACCEPT_DEFAULTS=1 ;;
        --no-lock)         RUN_LOCK=0 ;;
        --no-sync)         RUN_SYNC=0 ;;
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
        macos)            echo "macos" ;;
        windows)          [[ "$HAS_NVIDIA" == 1 ]] && echo "win-cuda"   || echo "cpu" ;;
        linux|wsl)        [[ "$HAS_NVIDIA" == 1 ]] && echo "linux-cuda" || echo "cpu" ;;
        *)                echo "cpu" ;;
    esac
}

if [[ "$TARGET" == "auto" ]]; then
    TARGET="$(auto_target)"
fi

# ---------- target -> template + suggested extras ----------
template_for_target() {
    case "$1" in
        cpu)        echo "pyproject_universal_cpu.toml" ;;
        linux-cuda) echo "pyproject_linux_cuda12.toml"  ;;
        win-cuda)   echo "pyproject_win_cuda12.toml"    ;;
        macos)      echo "pyproject_macos.toml"         ;;
        *)
            err "unknown target: $1"
            err "valid: cpu, linux-cuda, win-cuda, macos"
            exit 2
            ;;
    esac
}

suggested_extras_for_target() {
    case "$1" in
        cpu)        echo "" ;;                # no gpu extra on this template
        linux-cuda) echo "gpu" ;;
        win-cuda)   echo "gpu" ;;
        macos)      echo "" ;;
    esac
}

SRC="$(template_for_target "$TARGET")"
if [[ ! -f "$ROOT/$SRC" ]]; then
    err "template not found: $SRC"
    err "available templates:"
    ls "$ROOT"/pyproject_*.toml 2>/dev/null >&2 || true
    exit 1
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
printf '  %-18s %s\n' "Template:"         "$SRC"
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
        echo "Available targets: cpu, linux-cuda, win-cuda, macos"
        read -r -p "Pick target: " new_target
        TARGET="${new_target:-$TARGET}"
        SRC="$(template_for_target "$TARGET")"
        [[ -f "$ROOT/$SRC" ]] || { err "template not found: $SRC"; exit 1; }
        SUGGESTED_EXTRAS="$(suggested_extras_for_target "$TARGET")"
        info "Switched to target=$TARGET, template=$SRC, suggested extras='$SUGGESTED_EXTRAS'"
    fi
fi

# ---------- extras selection ----------
# Available extras per template (everything defined in [project.optional-dependencies]):
#   cpu:        sam, fifa, upscaler, dev
#   linux-cuda: gpu, sam, fifa, upscaler, dev
#   win-cuda:   gpu, sam, fifa, upscaler, dev
#   macos:      sam, fifa, upscaler, dev
AVAILABLE_EXTRAS_CPU="sam fifa upscaler dev"
AVAILABLE_EXTRAS_CUDA="gpu sam fifa upscaler dev"
case "$TARGET" in
    linux-cuda|win-cuda) AVAILABLE_EXTRAS="$AVAILABLE_EXTRAS_CUDA" ;;
    *)                   AVAILABLE_EXTRAS="$AVAILABLE_EXTRAS_CPU" ;;
esac

if [[ -n "$EXTRAS_CLI" ]]; then
    EXTRAS="$(echo "$EXTRAS_CLI" | tr ',' ' ' | tr -s ' ')"
elif [[ "$NON_INTERACTIVE" == 1 || "$ACCEPT_DEFAULTS" == 1 ]]; then
    EXTRAS="$SUGGESTED_EXTRAS"
else
    echo ""
    info "Available extras for $TARGET: $AVAILABLE_EXTRAS"
    echo "  gpu      = tensorrt + nvidia-ml-py (CUDA only)"
    echo "  sam      = SAM 3 video segmentation (sam3==0.1.3; CUDA at runtime)"
    echo "  fifa     = FIFA Skeletal Tracking Light (pytorch-lightning, timm, ...)"
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
    warn "Ignoring extras not defined in $SRC: $INVALID_EXTRAS"
fi
EXTRAS="$(echo "$VALID_EXTRAS" | tr -s ' ' | sed 's/^ //;s/ $//')"

# ---------- apply template ----------
echo ""
info "Copying template: $SRC -> pyproject.toml"
cp "$ROOT/$SRC" "$ROOT/pyproject.toml"

# ---------- lock ----------
if [[ "$RUN_LOCK" == 1 ]]; then
    if ! command -v uv >/dev/null 2>&1; then
        err "uv not found in PATH. Install uv: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    info "Running: uv lock"
    uv lock
else
    info "Skipping 'uv lock' (--no-lock)"
fi

# ---------- sync ----------
if [[ "$RUN_SYNC" == 1 ]]; then
    SYNC_CMD=(uv sync)
    for e in $EXTRAS; do
        SYNC_CMD+=(--extra "$e")
    done
    info "Running: ${SYNC_CMD[*]}"
    "${SYNC_CMD[@]}"
    ok ""
    ok "Done. vailá ready for target='$TARGET' with extras=[$EXTRAS]."
    say "Run the GUI:   uv run vaila.py"
else
    info "Skipping 'uv sync' (--no-sync)"
    say ""
    say "Next, run manually:"
    if [[ -n "$EXTRAS" ]]; then
        say "  uv sync $(echo "$EXTRAS" | sed 's/[^ ][^ ]*/--extra &/g')"
    else
        say "  uv sync"
    fi
fi
