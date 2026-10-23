#!/bin/sh
set -e

REMOTE_SIZE_CONNECT_TIMEOUT=3
REMOTE_SIZE_MAX_TIME=5
RELEASE_LOOKUP_MAX_TIME=15
DEFAULT_MODEL="Qwen/Qwen3-4B"
DEFAULT_QUANTIZATION=4
DEFAULT_UI_URL="http://localhost:1234/ui"
KEY_ESCAPE_TIMEOUT=1
MENU_FRAME_ROWS=4
PLAN_BASE_ROWS=8
INSTALL_STAGE_COUNT=4
PROGRESS_INTERVAL=0.2
PROGRESS_LOG_TAIL_BYTES=8192
PROGRESS_DEFAULT_COLUMNS=80
PROGRESS_LINE_PADDING=4
KEY_ESCAPE=$(printf '\033')
KEY_RETURN=$(printf '\r')
KEY_NEWLINE='
'
KEY_EOF=$(printf '\004')
KEY_INTERRUPT=$(printf '\003')
KEY_TAB=$(printf '\t')
INSTALL_TTY_STATE=""
INSTALL_TTY_OPEN=0
INSTALL_CURSOR_HIDDEN=0
INSTALL_LOG=""
INSTALL_STAGE=""
INSTALL_STAGE_NUMBER=0
PROGRESS_PID=""
LOG_READER_PID=""
PROGRESS_DOWNLOAD_PATH=""
PROGRESS_DOWNLOAD_SIZE=""
DOWNLOAD_STAGE=""
SOURCE_STAGE=""
ACTIVATION_PENDING=0

INSTALL_BACKEND="auto"
INSTALL_METHOD="auto"
INSTALL_RELEASE="latest"
INSTALL_VIDEO=0
DRY_RUN=0
INSTALL_VERBOSE=0

can_prompt() {
    [ -t 0 ] || ( : </dev/tty ) 2>/dev/null
}

read_answer() {
    if [ -t 0 ]; then
        IFS= read -r REPLY || error "Input closed. Installation cancelled."
    elif can_prompt; then
        IFS= read -r REPLY </dev/tty || error "Input closed. Installation cancelled."
    else
        error "No interactive terminal. Use --yes to accept the installation plan."
    fi
}

read_input() {
    if [ "${MISTRALRS_INSTALL_YES:-}" = "1" ]; then
        REPLY="y"
    else
        select_menu "${1:-Continue?}" y 'y:Yes' 'n:No'
        REPLY="$MENU_CHOICE"
    fi
}

RED=''
GREEN=''
YELLOW=''
BLUE=''
CYAN=''
DIM=''
BOLD=''
NC=''
MENU_REDRAW=0
if [ -t 1 ] && [ "${TERM:-dumb}" != "dumb" ]; then
    MENU_REDRAW=1
    if [ -z "${NO_COLOR+x}" ]; then
        RED='\033[0;31m'
        GREEN='\033[0;32m'
        YELLOW='\033[0;33m'
        BLUE='\033[0;34m'
        CYAN='\033[0;36m'
        DIM='\033[2m'
        BOLD='\033[1m'
        NC='\033[0m'
    fi
fi

restore_terminal() {
    if [ "$INSTALL_TTY_OPEN" = "1" ]; then
        if [ -n "$INSTALL_TTY_STATE" ]; then
            stty "$INSTALL_TTY_STATE" <&3 2>/dev/null || true
            INSTALL_TTY_STATE=""
        fi
        exec 3<&-
        INSTALL_TTY_OPEN=0
    fi
    if [ "$INSTALL_CURSOR_HIDDEN" = "1" ]; then
        printf '\033[?25h'
        INSTALL_CURSOR_HIDDEN=0
    fi
}

start_key_input() {
    if [ -t 0 ]; then
        exec 3<&0
    else
        exec 3</dev/tty
    fi
    INSTALL_TTY_OPEN=1
    INSTALL_TTY_STATE=$(stty -g <&3) || error "Could not read terminal settings."
    trap cleanup_installer 0
    trap 'exit 130' INT
    trap 'exit 143' TERM
    trap 'exit 129' HUP
    # Read Ctrl+C as a key so interrupts cannot land between successive key-reader processes.
    stty -echo -icanon -isig min 1 time 0 <&3 || error "Could not configure terminal input."
    printf '\033[?25l'
    INSTALL_CURSOR_HIDDEN=1
}

read_key_byte() {
    key_data=$(dd bs=1 count=1 <&3 2>/dev/null && printf '.') \
        || error "Could not read terminal input."
    KEY_BYTE=${key_data%.}
}

read_key() {
    read_key_byte
    case "$KEY_BYTE" in
        '') error "Input closed. Installation cancelled." ;;
        "$KEY_INTERRUPT") exit 130 ;;
        "$KEY_NEWLINE"|"$KEY_RETURN"|' ') KEY="enter" ;;
        "$KEY_EOF") KEY="escape" ;;
        "$KEY_TAB") KEY="down" ;;
        "$KEY_ESCAPE")
            KEY="escape"
            stty min 0 time "$KEY_ESCAPE_TIMEOUT" <&3
            read_key_byte
            case "$KEY_BYTE" in
                '['|'O')
                    KEY="ignore"
                    while :; do
                        read_key_byte
                        case "$KEY_BYTE" in
                            A) KEY="up"; break ;;
                            B) KEY="down"; break ;;
                            H) KEY="home"; break ;;
                            F) KEY="end"; break ;;
                            [0-9]|';'|'?') ;;
                            *) break ;;
                        esac
                    done
                    ;;
                '') ;;
                *) KEY="ignore" ;;
            esac
            stty min 1 time 0 <&3
            ;;
        *) KEY=$(printf '%s' "$KEY_BYTE" | LC_ALL=C tr '[:upper:]' '[:lower:]') ;;
    esac
}

erase_rows() {
    [ "$MENU_REDRAW" = "1" ] || return 0
    printf '\033[%sA\r\033[J' "$1"
}

render_menu() {
    printf "  ${BOLD}%s${NC}\n\n" "$menu_title"
    menu_row=1
    for menu_entry in "$@"; do
        if [ "$menu_row" = "$menu_index" ]; then
            printf "  ${CYAN}${BOLD}> [%s] %s${NC}\n" "${menu_entry%%:*}" "${menu_entry#*:}"
        else
            printf "    ${DIM}[%s]${NC} %s\n" "${menu_entry%%:*}" "${menu_entry#*:}"
        fi
        menu_row=$((menu_row + 1))
    done
    printf "\n  ${DIM}Up/Down to move, Enter to select, or press a shortcut.${NC}\n"
}

select_menu() {
    menu_title="$1"
    menu_default="$2"
    shift 2
    menu_count=$#
    menu_index=1
    menu_row=1
    menu_back="q"
    for menu_entry in "$@"; do
        [ "${menu_entry%%:*}" != "$menu_default" ] || menu_index=$menu_row
        [ "${menu_entry%%:*}" != "b" ] || menu_back="b"
        [ "${menu_entry%%:*}" != "n" ] || menu_back="n"
        menu_row=$((menu_row + 1))
    done
    MENU_CHOICE=""
    if [ "$MENU_REDRAW" != "1" ]; then
        printf '  %s\n\n' "$menu_title"
        for menu_entry in "$@"; do
            printf '  [%s] %s\n' "${menu_entry%%:*}" "${menu_entry#*:}"
        done
        while :; do
            printf '\n  Choice [%s]: ' "$menu_default"
            read_answer
            REPLY=${REPLY:-$menu_default}
            for menu_entry in "$@"; do
                [ "$REPLY" != "${menu_entry%%:*}" ] || MENU_CHOICE="$REPLY"
            done
            [ -z "$MENU_CHOICE" ] || return
            warn "Choose one of the listed shortcuts."
        done
    fi
    start_key_input
    render_menu "$@"
    while [ -z "$MENU_CHOICE" ]; do
        read_key
        case "$KEY" in
            up|k)
                menu_index=$((menu_index - 1))
                [ "$menu_index" -ge 1 ] || menu_index=$menu_count
                ;;
            down|j)
                menu_index=$((menu_index + 1))
                [ "$menu_index" -le "$menu_count" ] || menu_index=1
                ;;
            home) menu_index=1 ;;
            end) menu_index=$menu_count ;;
            enter)
                menu_row=1
                for menu_entry in "$@"; do
                    [ "$menu_row" != "$menu_index" ] || MENU_CHOICE="${menu_entry%%:*}"
                    menu_row=$((menu_row + 1))
                done
                ;;
            escape|q) MENU_CHOICE="$menu_back" ;;
            *)
                for menu_entry in "$@"; do
                    [ "$KEY" != "${menu_entry%%:*}" ] || MENU_CHOICE="$KEY"
                done
                ;;
        esac
        erase_rows "$((menu_count + MENU_FRAME_ROWS))"
        [ -n "$MENU_CHOICE" ] || render_menu "$@"
    done
    restore_terminal
}

# Print functions (output to stderr so they don't get captured in command substitution)
info() { printf "${BLUE}info:${NC} %s\n" "$1" >&2; }
success() { printf "${GREEN}success:${NC} %s\n" "$1" >&2; }
warn() { printf "${YELLOW}warning:${NC} %s\n" "$1" >&2; }
error() {
    printf "${RED}error:${NC} %s\n" "$1" >&2
    [ -z "$INSTALL_LOG" ] || printf 'error: %s\n' "$1" >>"$INSTALL_LOG"
    exit 1
}
# Echo a dependency-install command (rustup, brew, apt, ...) so the user sees it before confirming.
show_cmd() { printf "${BOLD}  \$ %s${NC}\n" "$1" >&2; }

stop_progress() {
    if [ -n "$PROGRESS_PID" ]; then
        kill "$PROGRESS_PID" 2>/dev/null || true
        wait "$PROGRESS_PID" 2>/dev/null || true
        PROGRESS_PID=""
        printf '\r\033[K'
    fi
    if [ "$INSTALL_CURSOR_HIDDEN" = "1" ]; then
        printf '\033[?25h'
        INSTALL_CURSOR_HIDDEN=0
    fi
}

cleanup_installer() {
    cleanup_status=$?
    set +e
    trap - 0
    stop_progress
    restore_terminal
    if [ -n "$LOG_READER_PID" ]; then
        kill "$LOG_READER_PID" 2>/dev/null || true
        wait "$LOG_READER_PID" 2>/dev/null || true
        LOG_READER_PID=""
    fi
    rollback_activation
    if [ -n "$INSTALL_LOG" ]; then
        rm -f "$INSTALL_LOG.pipe"
        if [ -f "$INSTALL_LOG.part" ]; then
            cat "$INSTALL_LOG.part" >>"$INSTALL_LOG"
            rm -f "$INSTALL_LOG.part"
        fi
        if [ "$cleanup_status" -eq 0 ]; then
            rm -f "$INSTALL_LOG"
        else
            printf '\n' >&2
            if [ "$cleanup_status" -eq 130 ]; then
                printf '  Installation cancelled.\n' >&2
            elif [ -n "$INSTALL_STAGE" ]; then
                printf "  ${RED}Installation stopped during: %s.${NC}\n" "$INSTALL_STAGE" >&2
            fi
            printf '  Installation log: %s\n' "$INSTALL_LOG" >&2
        fi
    fi
    [ -z "$DOWNLOAD_STAGE" ] || rm -rf "$DOWNLOAD_STAGE"
    [ -z "$SOURCE_STAGE" ] || rm -rf "$SOURCE_STAGE"
    [ -z "$SOURCE_INSTALL_ROOT" ] || rm -rf "$SOURCE_INSTALL_ROOT"
    exit "$cleanup_status"
}

begin_stage() {
    INSTALL_STAGE_NUMBER="$1"
    INSTALL_STAGE="$2"
    printf "\n  ${BOLD}[%s/%s] %s${NC}\n" "$INSTALL_STAGE_NUMBER" "$INSTALL_STAGE_COUNT" "$INSTALL_STAGE"
    printf '\n[%s/%s] %s\n' "$INSTALL_STAGE_NUMBER" "$INSTALL_STAGE_COUNT" "$INSTALL_STAGE" >>"$INSTALL_LOG"
}

stage_detail() {
    printf "    ${DIM}%s${NC}\n" "$1"
}

stage_done() {
    printf "    ${GREEN}[ok]${NC} %s\n" "$1"
}

elapsed_time() {
    elapsed_seconds=$(( $(date +%s) - PROGRESS_STARTED ))
    printf '%dm %02ds' "$((elapsed_seconds / 60))" "$((elapsed_seconds % 60))"
}

plain_log() {
    # ANSI escape ranges describe bytes, not locale-dependent character ordering.
    LC_ALL=C tr '\r' '\n' | LC_ALL=C sed "s/${KEY_ESCAPE}\[[0-?]*[ -/]*[@-~]//g"
}

show_failure_details() {
    plain_log <"$INSTALL_LOG.part" | LC_ALL=C awk '
        /^[[:space:]]*Building[[:space:]]+\[[^]]*\][[:space:]]+[0-9]+\/[0-9]+/ { next }
        { printf "      %s\n", $0 }' >&2
}

progress_detail() {
    if [ -n "$PROGRESS_DOWNLOAD_PATH" ]; then
        downloaded_bytes=0
        if [ -f "$PROGRESS_DOWNLOAD_PATH" ]; then
            downloaded_bytes=$(wc -c <"$PROGRESS_DOWNLOAD_PATH" | tr -d '[:space:]')
        fi
        if [ -n "$PROGRESS_DOWNLOAD_SIZE" ]; then
            downloaded_percent=$((downloaded_bytes * 100 / PROGRESS_DOWNLOAD_SIZE))
            [ "$downloaded_percent" -le 100 ] || downloaded_percent=100
            printf '%s%%  %s / %s' "$downloaded_percent" "$(human_size "$downloaded_bytes")" "$(human_size "$PROGRESS_DOWNLOAD_SIZE")"
        else
            printf '%s downloaded' "$(human_size "$downloaded_bytes")"
        fi
    else
        tail -c "$PROGRESS_LOG_TAIL_BYTES" "$INSTALL_LOG.part" 2>/dev/null | plain_log | LC_ALL=C awk '
            NF {
                sub(/^[[:space:]]+/, "")
                line = $0
                if ($0 ~ /^Building[[:space:]]+\[[^]]*\][[:space:]]+[0-9]+\/[0-9]+/) {
                    sub(/^Building[[:space:]]+\[[^]]*\][[:space:]]+/, "Building ")
                    build = $0
                }
            }
            END { print (build != "" ? build : line) }'
    fi
}

animate_progress() {
    trap - 0
    trap 'exit 0' INT TERM HUP
    progress_tick=0
    while :; do
        case "$((progress_tick % 4))" in
            0) progress_frame='|' ;;
            1) progress_frame='/' ;;
            2) progress_frame='-' ;;
            3) progress_frame='\' ;;
        esac
        current_activity=$(progress_detail)
        progress_text="[$progress_frame] $(elapsed_time)  ${current_activity:-$task_label}"
        printf "\r\033[K    ${CYAN}%.*s${NC}" "$PROGRESS_COLUMNS" "$progress_text"
        progress_tick=$((progress_tick + 1))
        sleep "$PROGRESS_INTERVAL"
    done
}

run_logged() {
    task_label="$1"
    shift
    : >"$INSTALL_LOG.part" || error "Could not create the command log. Check available disk space and permissions."
    printf '\n%s\n' "$task_label" >>"$INSTALL_LOG"
    PROGRESS_STARTED=$(date +%s)
    task_output="$INSTALL_LOG.part"
    if [ "$INSTALL_VERBOSE" = "1" ]; then
        stage_detail "$task_label..."
        task_output="$INSTALL_LOG.pipe"
        mkfifo "$task_output" || error "Could not start live logging. Check available disk space and permissions."
        tee "$INSTALL_LOG.part" <"$task_output" &
        LOG_READER_PID=$!
    elif [ "$MENU_REDRAW" = "1" ]; then
        PROGRESS_COLUMNS=$(stty size <&1 2>/dev/null | awk '{ print $2 }')
        PROGRESS_COLUMNS=${PROGRESS_COLUMNS:-$PROGRESS_DEFAULT_COLUMNS}
        [ "$PROGRESS_COLUMNS" -gt "$PROGRESS_LINE_PADDING" ] || PROGRESS_COLUMNS=$PROGRESS_DEFAULT_COLUMNS
        PROGRESS_COLUMNS=$((PROGRESS_COLUMNS - PROGRESS_LINE_PADDING - 1))
        printf '\033[?25l'
        INSTALL_CURSOR_HIDDEN=1
        animate_progress &
        PROGRESS_PID=$!
    else
        stage_detail "$task_label..."
    fi
    # Keep commands in the foreground so Ctrl+C also reaches their child processes.
    if "$@" >"$task_output" 2>&1; then
        task_status=0
    else
        task_status=$?
    fi
    stop_progress
    if [ -n "$LOG_READER_PID" ]; then
        if wait "$LOG_READER_PID"; then
            :
        else
            warn "Could not stream the complete command log."
            [ "$task_status" -ne 0 ] || task_status=1
        fi
        LOG_READER_PID=""
        rm -f "$INSTALL_LOG.pipe"
        printf '\n'
    fi
    cat "$INSTALL_LOG.part" >>"$INSTALL_LOG"
    if [ "$task_status" -eq 0 ]; then
        printf "    ${GREEN}[ok]${NC} %s ${DIM}(%s)${NC}\n" "$task_label" "$(elapsed_time)"
    else
        printf "    ${RED}[failed]${NC} %s ${DIM}(%s)${NC}\n" "$task_label" "$(elapsed_time)" >&2
        [ "$INSTALL_VERBOSE" = "1" ] || show_failure_details
    fi
    rm -f "$INSTALL_LOG.part"
    return "$task_status"
}

# Format a byte count for humans (e.g. 1234567 -> 1.2 MiB).
human_size() {
    awk -v b="$1" 'BEGIN {
        split("B KiB MiB GiB TiB", u, " ")
        i = 1
        while (b >= 1024 && i < 5) { b /= 1024; i++ }
        printf (i == 1 ? "%d %s" : "%.1f %s"), b, u[i]
    }'
}

# Content-Length of a URL after redirects; empty if it cannot be determined.
remote_download_size() {
    curl --proto '=https' --tlsv1.2 -sfIL \
        --connect-timeout "$REMOTE_SIZE_CONNECT_TIMEOUT" \
        --max-time "$REMOTE_SIZE_MAX_TIME" "$1" 2>/dev/null \
        | tr -d '\r' | awk 'tolower($1) == "content-length:" { len = $2 } END { if (len) print len }'
}

print_banner() {
    printf "\n${BOLD}"
    cat <<'EOF'
  __  __ _     _             _
 |  \/  (_)___| |_ _ __ __ _| |  _ __ ___
 | |\/| | / __| __| '__/ _` | | | '__/ __|
 | |  | | \__ \ |_| | | (_| | |_| |  \__ \
 |_|  |_|_|___/\__|_|  \__,_|_(_)_|  |___/
EOF
    printf "${NC}\n  ${BOLD}Install mistral.rs${NC}\n"
    printf "  ${BLUE}Fast, flexible LLM inference.${NC}\n\n"
}

print_help() {
    cat <<'EOF'
Usage: sh install.sh [options]
       curl -fsSL https://mistralrs.dev/install.sh | sh -s -- [options]

Run without options for guided installation.

  --release VERSION       latest (default), master, or a release tag such as v0.9.5
  --backend BACKEND       auto (default), cpu, metal, or cuda
  --from-source           Build the selected release from source
  --with-ffmpeg           Install optional FFmpeg video support
  --no-nccl               Omit NCCL from source builds
  --allow-cuda-mismatch   Allow a toolkit newer than the NVIDIA driver supports
  --yes, -y              Accept the plan and prerequisite installation prompts
  --dry-run              Preview the plan without downloading or changing files
  --verbose              Stream full command output while installing
  --help, -h             Show this help
EOF
}

load_options() {
    if [ -f "$PREBUILT_DIR/install.conf" ]; then
        while IFS='=' read -r option value; do
            case "$option" in
                release) INSTALL_RELEASE="$value" ;;
                backend) INSTALL_BACKEND="$value" ;;
                method) INSTALL_METHOD="$value" ;;
                no_nccl) MISTRALRS_INSTALL_NO_NCCL="${MISTRALRS_INSTALL_NO_NCCL:-$value}" ;;
                allow_cuda_mismatch)
                    MISTRALRS_INSTALL_ALLOW_CUDA_MISMATCH="${MISTRALRS_INSTALL_ALLOW_CUDA_MISMATCH:-$value}"
                    ;;
            esac
        done <"$PREBUILT_DIR/install.conf"
    fi
    if [ -n "${MISTRALRS_INSTALL_TAG:-}" ]; then
        INSTALL_RELEASE="$MISTRALRS_INSTALL_TAG"
    fi
    if [ -n "${MISTRALRS_INSTALL_FROM_SOURCE:-}" ]; then
        INSTALL_METHOD="source"
        INSTALL_RELEASE="${MISTRALRS_INSTALL_TAG:-master}"
    fi
}

save_options() {
    printf 'release=%s\nbackend=%s\nmethod=%s\nno_nccl=%s\nallow_cuda_mismatch=%s\n' \
        "$INSTALL_RELEASE" "$INSTALL_BACKEND" "$INSTALL_METHOD" \
        "${MISTRALRS_INSTALL_NO_NCCL:-0}" "${MISTRALRS_INSTALL_ALLOW_CUDA_MISMATCH:-0}" \
        >"$PREBUILT_DIR/install.conf"
}

validate_release() {
    case "$INSTALL_RELEASE" in
        ''|-*|*[!a-zA-Z0-9._-]*) return 1 ;;
    esac
    return 0
}

parse_args() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --help|-h) print_help; exit 0 ;;
            --yes|-y) MISTRALRS_INSTALL_YES=1 ;;
            --dry-run) DRY_RUN=1 ;;
            --verbose) INSTALL_VERBOSE=1 ;;
            --from-source) INSTALL_METHOD="source" ;;
            --with-ffmpeg) INSTALL_VIDEO=1 ;;
            --no-nccl) MISTRALRS_INSTALL_NO_NCCL=1 ;;
            --allow-cuda-mismatch) MISTRALRS_INSTALL_ALLOW_CUDA_MISMATCH=1 ;;
            --release|--backend)
                [ "$#" -ge 2 ] || error "Missing value for $1. See --help."
                case "$1" in
                    --release) INSTALL_RELEASE="$2" ;;
                    --backend) INSTALL_BACKEND="$2" ;;
                esac
                shift
                ;;
            *) error "Unknown option: $1. See --help." ;;
        esac
        shift
    done
    validate_release || error "Invalid release: $INSTALL_RELEASE"
    case "$INSTALL_BACKEND" in
        auto|cpu|metal|cuda) ;;
        *) error "Unknown backend: $INSTALL_BACKEND. Choose auto, cpu, metal, or cuda." ;;
    esac
    case "$INSTALL_METHOD" in
        auto|source) ;;
        *) error "Unknown installation method: $INSTALL_METHOD." ;;
    esac
}

resolve_plan() {
    SELECTED_BACKEND="$INSTALL_BACKEND"
    if [ "$SELECTED_BACKEND" = "auto" ]; then
        if [ "$INSTALL_OS" = "macos" ] && [ "$INSTALL_ARCH" = "arm64" ]; then
            SELECTED_BACKEND="metal"
        elif [ "$INSTALL_OS" = "linux" ] && [ -n "$DETECTED_CUDA_CC" ]; then
            SELECTED_BACKEND="cuda"
        else
            SELECTED_BACKEND="cpu"
        fi
    fi
    case "$SELECTED_BACKEND" in
        metal)
            [ "$INSTALL_OS" = "macos" ] && [ "$INSTALL_ARCH" = "arm64" ] \
                || error "Metal installation requires Apple Silicon macOS."
            ;;
        cuda)
            [ "$INSTALL_OS" = "linux" ] && [ -n "$DETECTED_CUDA_CC" ] \
                || error "CUDA installation requires Linux and a detected NVIDIA GPU."
            ;;
    esac
    PLANNED_ASSET=""
    PLANNED_METHOD="source"
    if [ "$INSTALL_METHOD" != "source" ] && [ "$INSTALL_RELEASE" != "master" ]; then
        PLANNED_ASSET=$(detect_prebuilt_asset "$INSTALL_OS")
        [ -z "$PLANNED_ASSET" ] || PLANNED_METHOD="prebuilt"
    fi
    case "$INSTALL_RELEASE" in
        latest) RELEASE_LABEL="Latest stable" ;;
        master) RELEASE_LABEL="Development (master)" ;;
        *) RELEASE_LABEL="$INSTALL_RELEASE" ;;
    esac
    case "$SELECTED_BACKEND" in
        metal) BACKEND_LABEL="Metal" ;;
        cuda) BACKEND_LABEL="NVIDIA CUDA" ;;
        cpu) BACKEND_LABEL="CPU" ;;
    esac
}

plan_row() {
    printf "  ${DIM}%-18s${NC} %s\n" "$1" "$2"
}

print_plan() {
    PLAN_ROWS="$PLAN_BASE_ROWS"
    plan_row "Platform" "$INSTALL_OS ($INSTALL_ARCH)"
    plan_row "Version" "$RELEASE_LABEL"
    if [ "$INSTALL_BACKEND" = "auto" ]; then
        printf "  ${DIM}%-18s${NC} ${GREEN}%s (recommended)${NC}\n" "Acceleration" "$BACKEND_LABEL"
    else
        plan_row "Acceleration" "$BACKEND_LABEL"
    fi
    if [ "$PLANNED_METHOD" = "prebuilt" ]; then
        plan_row "Installation" "Prebuilt binary"
    else
        plan_row "Installation" "Build from source"
    fi
    plan_row "Command location" "$(tildify "$BIN_DIR")"
    plan_row "Managed files" "$(tildify "$PREBUILT_DIR")"
    if [ "$INSTALL_VIDEO" = "1" ]; then
        plan_row "Video support" "Install FFmpeg if needed"
    else
        plan_row "Video support" "Leave FFmpeg unchanged"
    fi
    if [ "$PLANNED_METHOD" = "source" ]; then
        printf "\n  ${YELLOW}A source build needs Rust and platform build tools.${NC}\n"
        PLAN_ROWS=$((PLAN_ROWS + 2))
    fi
    printf '\n'
}

customize_plan() {
    while :; do
        select_menu 'Customize installation' 1 \
            "1:Version: $INSTALL_RELEASE" "2:Acceleration: $INSTALL_BACKEND" \
            "3:Installation method: $INSTALL_METHOD" "4:Video support" 'b:Back'
        case "$MENU_CHOICE" in
            b) return ;;
            1)
                case "$INSTALL_RELEASE" in
                    latest) version_default=l ;;
                    master) version_default=d ;;
                    *) version_default=t ;;
                esac
                select_menu 'Version' "$version_default" \
                    'l:Latest stable (recommended)' 'd:Development (master)' 't:Specific release tag' 'b:Back'
                case "$MENU_CHOICE" in
                    l) INSTALL_RELEASE="latest" ;;
                    d) INSTALL_RELEASE="master" ;;
                    t)
                        printf '  Release tag (for example v0.9.5) [%s]: ' "$INSTALL_RELEASE"
                        read_answer
                        previous_release="$INSTALL_RELEASE"
                        [ -z "$REPLY" ] || INSTALL_RELEASE="$REPLY"
                        if ! validate_release; then
                            warn "Use a release tag such as v0.9.5."
                            INSTALL_RELEASE="$previous_release"
                        fi
                        printf '\n'
                        ;;
                esac
                ;;
            2)
                case "$INSTALL_BACKEND" in
                    auto) backend_default=a ;;
                    cpu) backend_default=c ;;
                    metal) backend_default=m ;;
                    cuda) backend_default=g ;;
                esac
                set -- 'a:Automatic (recommended)' 'c:CPU only'
                if [ "$INSTALL_OS" = "macos" ] && [ "$INSTALL_ARCH" = "arm64" ]; then
                    set -- "$@" 'm:Metal'
                elif [ "$INSTALL_OS" = "linux" ] && [ -n "$DETECTED_CUDA_CC" ]; then
                    set -- "$@" 'g:NVIDIA CUDA'
                fi
                select_menu 'Acceleration' "$backend_default" "$@" 'b:Back'
                case "$MENU_CHOICE" in
                    a) INSTALL_BACKEND="auto" ;;
                    c) INSTALL_BACKEND="cpu" ;;
                    m) INSTALL_BACKEND="metal" ;;
                    g) INSTALL_BACKEND="cuda" ;;
                esac
                ;;
            3)
                method_default=p
                [ "$INSTALL_METHOD" != "source" ] || method_default=s
                select_menu 'Installation method' "$method_default" \
                    'p:Prebuilt when available (recommended)' 's:Build from source' 'b:Back'
                case "$MENU_CHOICE" in
                    p) INSTALL_METHOD="auto" ;;
                    s) INSTALL_METHOD="source" ;;
                esac
                ;;
            4)
                video_default=n
                [ "$INSTALL_VIDEO" != "1" ] || video_default=y
                select_menu 'Video support' "$video_default" \
                    'n:Leave FFmpeg unchanged' 'y:Install FFmpeg if needed' 'b:Back'
                case "$MENU_CHOICE" in
                    y) INSTALL_VIDEO=1 ;;
                    n) INSTALL_VIDEO=0 ;;
                esac
                ;;
        esac
    done
}

guided_install() {
    while :; do
        resolve_plan
        print_plan
        if [ "${MISTRALRS_INSTALL_YES:-}" = "1" ]; then
            return
        fi
        if ! can_prompt; then
            [ "$DRY_RUN" = "1" ] && return
            error "No interactive terminal. Use --yes to accept this plan."
        fi
        install_action='i:Install'
        [ "$DRY_RUN" != "1" ] || install_action='i:Preview'
        select_menu 'Installation' i "$install_action" 'c:Customize' 'q:Cancel'
        case "$MENU_CHOICE" in
            i) return ;;
            c)
                erase_rows "$PLAN_ROWS"
                customize_plan
                ;;
            q) printf '  Installation cancelled.\n'; exit 0 ;;
        esac
    done
}

configure_release() {
    MISTRALRS_INSTALL_TAG=""
    if [ "$INSTALL_RELEASE" = "latest" ] && [ "$PLANNED_METHOD" = "source" ]; then
        run_logged "Resolve latest stable release" curl --proto '=https' --tlsv1.2 -fsSLI \
            --connect-timeout "$REMOTE_SIZE_CONNECT_TIMEOUT" --max-time "$RELEASE_LOOKUP_MAX_TIME" \
            -o /dev/null -w '%{url_effective}' "$MISTRALRS_REPO_URL/releases/latest" \
            || error "Could not resolve the latest stable release. Retry or choose --release vVERSION."
        latest_url=$(tail -n 1 "$INSTALL_LOG")
        MISTRALRS_INSTALL_TAG=${latest_url##*/}
        case "$MISTRALRS_INSTALL_TAG" in
            v[0-9]*) ;;
            *) error "Could not resolve the latest stable release. Choose --release vVERSION." ;;
        esac
    elif [ "$INSTALL_RELEASE" != "latest" ] && [ "$INSTALL_RELEASE" != "master" ]; then
        MISTRALRS_INSTALL_TAG="$INSTALL_RELEASE"
    fi
    if [ -n "$MISTRALRS_INSTALL_TAG" ]; then
        RELEASE_BASE="$MISTRALRS_REPO_URL/releases/download/$MISTRALRS_INSTALL_TAG"
    else
        RELEASE_BASE="$MISTRALRS_REPO_URL/releases/latest/download"
    fi
}

# Detect operating system
detect_os() {
    case "$(uname -s)" in
        Darwin*)
            echo "macos"
            ;;
        Linux*)
            echo "linux"
            ;;
        *)
            error "Unsupported operating system: $(uname -s)"
            ;;
    esac
}

# Minimum required Rust version
REQUIRED_RUST_VERSION="1.94"
RUSTUP_INSTALL_CMD="curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
MISTRALRS_REPO_URL="https://github.com/EricLBuehler/mistral.rs"
MISTRALRS_BRANCH="master"
MISTRALRS_CLI_PACKAGE="mistralrs-cli"

# Check if Rust is installed
check_rust() {
    command -v cargo >/dev/null 2>&1
}

# Get installed Rust version (major.minor)
get_rust_version() {
    rustc --version 2>/dev/null | sed -n 's/rustc \([0-9]*\.[0-9]*\).*/\1/p'
}

# Compare two version strings (returns 0 if $1 >= $2, 1 otherwise)
version_gte() {
    v1_major=$(echo "$1" | cut -d. -f1)
    v1_minor=$(echo "$1" | cut -d. -f2)
    v2_major=$(echo "$2" | cut -d. -f1)
    v2_minor=$(echo "$2" | cut -d. -f2)

    if [ "$v1_major" -gt "$v2_major" ] 2>/dev/null; then
        return 0
    elif [ "$v1_major" -eq "$v2_major" ] 2>/dev/null && [ "$v1_minor" -ge "$v2_minor" ] 2>/dev/null; then
        return 0
    fi
    return 1
}

# Install Rust via rustup
install_rust() {
    run_logged "Install Rust" sh -c "$RUSTUP_INSTALL_CMD" \
        || error "Rust installation failed. Check the installation log, then retry."
    . "$HOME/.cargo/env"
}

# Update Rust to latest version
update_rust() {
    run_logged "Update Rust" rustup update stable \
        || error "Rust update failed. Check the installation log, then retry."
}

# Detect CUDA compute capability
detect_cuda_compute_cap() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo ""
        return
    fi

    # Try direct query
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')

    if [ -n "$cc" ]; then
        echo "$cc"
    fi
}

# Detect CUDA toolkit major version from nvcc (e.g. 13). Empty if nvcc is unavailable.
# CUDA toolkit version as major*100+minor (e.g. 13.1 -> 1301), empty if nvcc absent.
detect_cuda_version_code() {
    if command -v nvcc >/dev/null 2>&1; then
        ver=$(nvcc --version 2>/dev/null | grep -oE "release [0-9]+\.[0-9]+" | head -1 | grep -oE "[0-9]+\.[0-9]+")
        if [ -n "$ver" ]; then
            echo $(( ${ver%%.*} * 100 + ${ver#*.} ))
        fi
    fi
}

# Detect the maximum CUDA runtime version supported by the installed NVIDIA driver.
detect_cuda_driver_version_code() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo ""
        return
    fi

    ver=$(nvidia-smi 2>/dev/null | grep -oE "CUDA.*Version:[[:space:]]*[0-9]+\.[0-9]+" | head -1 | grep -oE "[0-9]+\.[0-9]+")
    if [ -n "$ver" ]; then
        echo $(( ${ver%%.*} * 100 + ${ver#*.} ))
    fi
}

version_code_to_str() {
    code="$1"
    if [ -n "$code" ]; then
        printf "%s.%s" $(( code / 100 )) $(( code % 100 ))
    fi
}

cutile_supported_for_cuda() {
    cuda_ver_code="$1"
    cuda_cc="$2"

    [ -n "$cuda_ver_code" ] && [ -n "$cuda_cc" ] || return 1

    if [ "$cuda_cc" -ge 80 ] 2>/dev/null && [ "$cuda_cc" -lt 90 ] 2>/dev/null && [ "$cuda_ver_code" -ge 1302 ] 2>/dev/null; then
        return 0
    fi
    if [ "$cuda_cc" -eq 90 ] 2>/dev/null && [ "$cuda_ver_code" -ge 1303 ] 2>/dev/null; then
        return 0
    fi
    if [ "$cuda_cc" -ge 100 ] 2>/dev/null && [ "$cuda_ver_code" -ge 1302 ] 2>/dev/null; then
        return 0
    fi

    return 1
}

check_cuda_source_build_versions() {
    os="$1"
    [ "$os" = "linux" ] || return 0

    cuda_cc=$(detect_cuda_compute_cap)
    [ -n "$cuda_cc" ] || return 0

    cuda_ver_code=$(detect_cuda_version_code)
    driver_cuda_code=$(detect_cuda_driver_version_code)
    [ -n "$cuda_ver_code" ] && [ -n "$driver_cuda_code" ] || return 0

    if [ "$cuda_ver_code" -gt "$driver_cuda_code" ] 2>/dev/null; then
        if [ "${MISTRALRS_INSTALL_ALLOW_CUDA_MISMATCH:-}" = "1" ]; then
            warn "Local nvcc CUDA $(version_code_to_str "$cuda_ver_code") is newer than the NVIDIA driver supports ($(version_code_to_str "$driver_cuda_code")); continuing because MISTRALRS_INSTALL_ALLOW_CUDA_MISMATCH=1."
        else
            error "Local nvcc CUDA $(version_code_to_str "$cuda_ver_code") is newer than the NVIDIA driver supports ($(version_code_to_str "$driver_cuda_code")). Source builds can fail with CUDA_ERROR_UNSUPPORTED_PTX_VERSION; upgrade the driver, install a matching CUDA toolkit, use a prebuilt, or set MISTRALRS_INSTALL_ALLOW_CUDA_MISMATCH=1 to override."
        fi
    fi
}

# Check if MKL is installed
detect_mkl() {
    # Check MKLROOT environment variable
    if [ -n "$MKLROOT" ] && [ -d "$MKLROOT" ]; then
        return 0
    fi

    # Check common installation paths
    for path in /opt/intel/oneapi/mkl/latest /opt/intel/mkl /opt/intel/oneapi/mkl; do
        if [ -d "$path" ]; then
            return 0
        fi
    done

    return 1
}

# Check if CPU is Intel
is_intel_cpu() {
    if [ -f /proc/cpuinfo ]; then
        grep -qi "intel" /proc/cpuinfo && return 0
    elif command -v sysctl >/dev/null 2>&1; then
        sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -qi "intel" && return 0
    fi
    return 1
}

# Check/install Xcode Command Line Tools (macOS)
check_xcode_cli_tools() {
    if ! xcrun --version >/dev/null 2>&1; then
        warn "Xcode Command Line Tools are not installed"
        echo ""
        show_cmd "xcode-select --install"
        read_input "Install Xcode Command Line Tools?"
        case "$REPLY" in
            [Nn]*)
                error "Xcode Command Line Tools are required for Metal support"
                ;;
        esac
        info "Installing Xcode Command Line Tools..."
        xcode-select --install
        read_input "Continue after completing the Xcode installation dialog?"
        case "$REPLY" in
            [Nn]*) error "Installation cancelled." ;;
        esac
        show_cmd "sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer"
        sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
    fi
}

# Check/install Metal Toolchain (macOS)
check_metal_toolchain() {
    if ! xcrun metal --version >/dev/null 2>&1; then
        warn "Metal Toolchain is not installed"
        echo ""
        show_cmd "xcodebuild -downloadComponent MetalToolchain"
        read_input "Install the Metal toolchain?"
        case "$REPLY" in
            [Nn]*)
                error "Metal Toolchain is required for Metal support"
                ;;
        esac
        info "Installing Metal Toolchain..."
        xcodebuild -downloadComponent MetalToolchain
    fi
}

# Check if cuDNN is installed
detect_cudnn() {
    # Check common cuDNN library paths
    for path in /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64 /usr/lib64; do
        if [ -f "$path/libcudnn.so" ] || ls "$path"/libcudnn.so.* >/dev/null 2>&1; then
            return 0
        fi
    done
    return 1
}

# Check if NCCL is installed
detect_nccl() {
    for root in "$NCCL_ROOT" "$NCCL_HOME" "$CUDA_HOME" "$CUDA_PATH" /usr/local/cuda; do
        [ -n "$root" ] || continue
        for subdir in lib lib64 lib/x86_64-linux-gnu; do
            if ls "$root/$subdir"/libnccl.so* >/dev/null 2>&1; then
                return 0
            fi
        done
    done

    if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q "libnccl\\.so"; then
        return 0
    fi

    for path in /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/local/lib /usr/local/lib64 /usr/lib64; do
        if ls "$path"/libnccl.so* >/dev/null 2>&1; then
            return 0
        fi
    done

    return 1
}

# Build feature string based on detected hardware
build_features() {
    os="$1"
    features=""

    if [ "$SELECTED_BACKEND" = "metal" ]; then
        features="metal"
    elif [ "$SELECTED_BACKEND" = "cuda" ]; then
        cuda_cc="$DETECTED_CUDA_CC"
        features="cuda"
        if [ "${MISTRALRS_INSTALL_NO_NCCL:-}" != "1" ]; then
            if detect_nccl; then
                features="$features nccl"
            elif [ "${MISTRALRS_INSTALL_NCCL:-}" = "1" ]; then
                features="$features nccl"
                warn "MISTRALRS_INSTALL_NCCL=1 set but NCCL was not detected; the build may fail unless libnccl is on the linker path"
            else
                warn "NCCL not found - skipping nccl. Install NCCL or set MISTRALRS_INSTALL_NCCL=1 to force it; NCCL is the preferred CUDA multi-GPU path."
            fi
        fi

        if detect_cudnn; then
            features="$features cudnn"
        fi

        if [ "$cuda_cc" = "90" ]; then
            features="$features flash-attn-v3 flash-attn"
        elif [ "$cuda_cc" -ge 80 ] 2>/dev/null; then
            features="$features flash-attn"
        fi

        cuda_ver_code=$(detect_cuda_version_code)
        if cutile_supported_for_cuda "$cuda_ver_code" "$cuda_cc"; then
            features="$features cutile"
        fi
    fi

    # Check for MKL on Intel
    if is_intel_cpu && detect_mkl; then
        features="$features mkl"
    fi

    # Trim leading/trailing whitespace
    echo "$features" | xargs
}

# Check if ffmpeg is installed
check_ffmpeg() {
    command -v ffmpeg >/dev/null 2>&1 && ffmpeg -version >/dev/null 2>&1
}

# The package-manager command that would install FFmpeg on this system; empty if none was detected.
ffmpeg_install_cmd() {
    os="$1"
    if [ "$os" = "macos" ]; then
        if command -v brew >/dev/null 2>&1; then
            if brew list ffmpeg >/dev/null 2>&1; then
                echo "brew reinstall ffmpeg"
            else
                echo "brew install ffmpeg"
            fi
        fi
    elif command -v apt-get >/dev/null 2>&1; then
        echo "sudo apt-get update && sudo apt-get install -y ffmpeg"
    elif command -v dnf >/dev/null 2>&1; then
        echo "sudo dnf install -y ffmpeg"
    fi
}

# Install ffmpeg using the system package manager
install_ffmpeg() {
    os="$1"
    cmd=$(ffmpeg_install_cmd "$os")
    if [ -z "$cmd" ]; then
        if [ "$os" = "macos" ]; then
            warn "Homebrew not found. Install FFmpeg manually: https://ffmpeg.org/download.html"
        else
            warn "Could not detect package manager. Install FFmpeg manually: https://ffmpeg.org/download.html"
        fi
        return 1
    fi
    info "Installing FFmpeg..."
    show_cmd "$cmd"
    sh -c "$cmd"
}

# Install mistralrs-cli
install_mistralrs() {
    features="$1"
    SOURCE_INSTALL_ROOT=$(mktemp -d)
    SOURCE_MISTRALRS="$SOURCE_INSTALL_ROOT/bin/mistralrs"

    if [ -n "$MISTRALRS_INSTALL_TAG" ]; then
        set -- --tag "$MISTRALRS_INSTALL_TAG"
        ref_desc="tag $MISTRALRS_INSTALL_TAG"
    else
        set -- --branch "$MISTRALRS_BRANCH"
        ref_desc="branch $MISTRALRS_BRANCH"
    fi

    stage_detail "Source: $ref_desc"
    stage_detail "The first build can take several minutes."
    [ -z "$features" ] || set -- "$@" --features "$features"
    cargo_progress=never
    if [ "$MENU_REDRAW" = "1" ] && [ "$INSTALL_VERBOSE" = "0" ]; then
        cargo_progress=always
    fi
    run_logged "Compile mistral.rs ($BACKEND_LABEL)" \
        env CARGO_TERM_COLOR=never CARGO_TERM_PROGRESS_WHEN="$cargo_progress" \
        CARGO_TERM_PROGRESS_WIDTH="$PROGRESS_DEFAULT_COLUMNS" CARGO_TERM_PROGRESS_TERM_INTEGRATION=false \
        CARGO_TERM_UNICODE=false CARGO_TERM_HYPERLINKS=false \
        cargo install --root "$SOURCE_INSTALL_ROOT" --force --locked \
        --git "$MISTRALRS_REPO_URL" "$@" "$MISTRALRS_CLI_PACKAGE" \
        || error "Source build failed. Check the compiler error and installation log, then retry. Existing installation kept."
}

remove_legacy_tileiras_link() {
    legacy_tileiras_link="$BIN_DIR/tileiras"
    legacy_tileiras_target="$PREBUILT_DIR/bin/tileiras"
    if [ -L "$legacy_tileiras_link" ] && [ "$(readlink "$legacy_tileiras_link" 2>/dev/null)" = "$legacy_tileiras_target" ]; then
        rm -f "$legacy_tileiras_link"
    fi
}

activate_install() {
    activation_stage="$1"
    activation_backup=""
    mkdir -p "$BIN_DIR" || return 1
    if [ -e "$PREBUILT_DIR" ]; then
        activation_backup=$(mktemp -d "$HOME/.mistralrs-backup.XXXXXX") || return 1
        rmdir "$activation_backup" || return 1
        ACTIVATION_PENDING=1
        mv "$PREBUILT_DIR" "$activation_backup" || return 1
    fi
    ACTIVATION_PENDING=1
    if mv "$activation_stage" "$PREBUILT_DIR"; then
        if "$PREBUILT_DIR/mistralrs" --version >/dev/null 2>&1 \
            && ln -sf "$PREBUILT_DIR/mistralrs" "$BIN_DIR/mistralrs"; then
            ACTIVATION_PENDING=0
            remove_legacy_tileiras_link
            [ -z "$activation_backup" ] || rm -rf "$activation_backup"
            return 0
        fi
    fi
    rollback_activation
    return 1
}

rollback_activation() {
    [ "$ACTIVATION_PENDING" = "1" ] || return 0
    if [ -n "$activation_backup" ] && { [ -e "$activation_backup" ] || [ -L "$activation_backup" ]; }; then
        if ! rm -rf "$PREBUILT_DIR" || ! mv "$activation_backup" "$PREBUILT_DIR"; then
            warn "Could not restore the previous installation. Its files remain at $activation_backup."
            return 1
        fi
    elif [ ! -e "$activation_stage" ]; then
        rm -rf "$PREBUILT_DIR" || return 1
    fi
    ACTIVATION_PENDING=0
}

stage_source_binary() {
    if [ ! -f "$SOURCE_MISTRALRS" ]; then
        error "cargo install succeeded but $SOURCE_MISTRALRS was not found"
    fi
    SOURCE_STAGE=$(mktemp -d "$HOME/.mistralrs-stage.XXXXXX")
    cp "$SOURCE_MISTRALRS" "$SOURCE_STAGE/mistralrs"
    INSTALL_CANDIDATE="$SOURCE_STAGE"
}

# Prebuilt binaries: SMs we publish a CUDA build for, per arch (see .github/workflows/release.yml).
# aarch64 covers the Grace parts only (GH200/GB200/GB10).
PREBUILT_CUDA_SMS_X86="80 86 89 90 100 120"
PREBUILT_CUDA_SMS_AARCH64="90 100 121"
# Newest first. Format is asset-token:minimum-driver-cuda-code.
PREBUILT_CUDA_VARIANTS="133:1303 132:1302 131:1301 130:1300 129:1209 128:1208"
RELEASE_BASE=""
PREBUILT_DIR="$HOME/.mistralrs"
BIN_DIR="$HOME/.local/bin"
CARGO_BIN_DIR="${CARGO_HOME:-$HOME/.cargo}/bin"
CARGO_MISTRALRS="$CARGO_BIN_DIR/mistralrs"
MISTRALRS_ENV="$PREBUILT_DIR/env"
REPLACE_DUPLICATE_INSTALLS=""
SOURCE_INSTALL_ROOT=""
SOURCE_MISTRALRS=""

cuda_sms_for_variant() {
    cuda_token="$1"
    arch="$2"

    case "$cuda_token:$arch" in
        128:x86_64) echo "80 86 89 90 100 120" ;;
        128:aarch64|128:arm64) echo "90 100" ;;
        129:aarch64|129:arm64) echo "121" ;;
        130:x86_64) echo "$PREBUILT_CUDA_SMS_X86" ;;
        130:aarch64|130:arm64) echo "$PREBUILT_CUDA_SMS_AARCH64" ;;
        131:x86_64) echo "$PREBUILT_CUDA_SMS_X86" ;;
        131:aarch64|131:arm64) echo "$PREBUILT_CUDA_SMS_AARCH64" ;;
        132:x86_64) echo "80 86 89 100 120" ;;
        132:aarch64|132:arm64) echo "100 121" ;;
        133:x86_64) echo "90" ;;
        133:aarch64|133:arm64) echo "90" ;;
    esac
}

# Echo the prebuilt asset name for this platform, or empty if none is published for it.
detect_prebuilt_asset() {
    os="$1"
    arch=$(uname -m)
    if [ "$os" = "macos" ]; then
        [ "$SELECTED_BACKEND" = "metal" ] && echo "mistralrs-metal-aarch64-apple-darwin.tar.gz"
        return 0
    fi
    # Linux x86_64 and aarch64 have prebuilts; other arches build from source.
    case "$arch" in
        x86_64) triple="x86_64-unknown-linux-gnu" ;;
        aarch64|arm64) triple="aarch64-unknown-linux-gnu" ;;
        *) return ;;
    esac
    cc=$(detect_cuda_compute_cap)
    if [ "$SELECTED_BACKEND" = "cuda" ]; then
        driver_cuda_code=$(detect_cuda_driver_version_code)
        if [ -z "$driver_cuda_code" ]; then
            warn "Could not detect the CUDA version supported by the NVIDIA driver; building from source."
            return 0
        fi

        for variant in $PREBUILT_CUDA_VARIANTS; do
            cuda_token=${variant%:*}
            cuda_min_code=${variant#*:}
            if [ "$driver_cuda_code" -ge "$cuda_min_code" ] 2>/dev/null; then
                cuda_sms=$(cuda_sms_for_variant "$cuda_token" "$arch")
                for sm in $cuda_sms; do
                    if [ "$cc" = "$sm" ]; then
                        echo "mistralrs-cuda${cuda_token}-sm${cc}-${triple}.tar.gz"
                        return 0
                    fi
                done
            fi
        done
        if [ "$driver_cuda_code" -lt 1208 ] 2>/dev/null; then
            warn "NVIDIA driver supports CUDA $(version_code_to_str "$driver_cuda_code"), but the oldest CUDA prebuilt requires CUDA 12.8; building from source."
        else
            warn "No CUDA prebuilt matches compute capability $cc and driver CUDA $(version_code_to_str "$driver_cuda_code"); building from source."
        fi
        return 0
    fi
    if [ "$triple" = "aarch64-unknown-linux-gnu" ]; then
        # the default aarch64 asset requires dotprod, fp16, and fhm
        if ! grep -qw asimddp /proc/cpuinfo 2>/dev/null \
            || ! grep -qw asimdhp /proc/cpuinfo 2>/dev/null \
            || ! grep -qw asimdfhm /proc/cpuinfo 2>/dev/null; then
            echo "mistralrs-cpu-${triple}-v8_0.tar.gz"
            return 0
        fi
    fi
    echo "mistralrs-cpu-${triple}.tar.gz"
}

detect_legacy_cuda_prebuilt_asset() {
    os="$1"
    arch=$(uname -m)
    [ "$os" = "linux" ] && [ "$SELECTED_BACKEND" = "cuda" ] || return 0
    case "$arch" in
        x86_64) triple="x86_64-unknown-linux-gnu"; cuda_sms="$PREBUILT_CUDA_SMS_X86" ;;
        aarch64|arm64) triple="aarch64-unknown-linux-gnu"; cuda_sms="$PREBUILT_CUDA_SMS_AARCH64" ;;
        *) return 0 ;;
    esac
    cc=$(detect_cuda_compute_cap)
    [ -n "$cc" ] || return 0
    driver_cuda_code=$(detect_cuda_driver_version_code)
    [ -n "$driver_cuda_code" ] && [ "$driver_cuda_code" -ge 1301 ] 2>/dev/null || return 0
    for sm in $cuda_sms; do
        if [ "$cc" = "$sm" ]; then
            echo "mistralrs-cuda-sm${cc}-${triple}.tar.gz"
            return 0
        fi
    done
    return 0
}

download_prebuilt() {
    asset="$1"
    asset_url="$RELEASE_BASE/$asset"
    download_size=$(remote_download_size "$asset_url")
    case "$download_size" in
        ''|*[!0-9]*|0) download_size="" ;;
    esac
    PROGRESS_DOWNLOAD_PATH="$DOWNLOAD_STAGE/$asset"
    PROGRESS_DOWNLOAD_SIZE="$download_size"
    download_label="Download $BACKEND_LABEL binary"
    [ -z "$download_size" ] || download_label="$download_label ($(human_size "$download_size"))"
    run_logged "$download_label" curl --proto '=https' --tlsv1.2 -fsSL \
        "$asset_url" -o "$PROGRESS_DOWNLOAD_PATH"
}

prepare_source_build() {
    os="$1"
    if [ "$SELECTED_BACKEND" = "cuda" ]; then
        command -v nvcc >/dev/null 2>&1 \
            || error "CUDA source builds require nvcc. Install the CUDA toolkit or choose --backend cpu."
        check_cuda_source_build_versions "$os"
    fi
    # Check for Rust
    if check_rust; then
        rust_version_full=$(rustc --version 2>/dev/null || echo "unknown")
        rust_version=$(get_rust_version)
        stage_detail "$rust_version_full"

        if [ -n "$rust_version" ] && ! version_gte "$rust_version" "$REQUIRED_RUST_VERSION"; then
            warn "Rust $rust_version is below the required version $REQUIRED_RUST_VERSION"
            echo ""
            show_cmd "rustup update stable"
            read_input "Update Rust?"
            case "$REPLY" in
                [Nn]*)
                    error "Rust $REQUIRED_RUST_VERSION or newer is required to install mistral.rs"
                    ;;
            esac
            update_rust
            rust_version=$(get_rust_version)
            if ! version_gte "$rust_version" "$REQUIRED_RUST_VERSION"; then
                error "Failed to update Rust to required version $REQUIRED_RUST_VERSION"
            fi
        fi
    else
        warn "Rust is not installed"
        echo ""
        show_cmd "$RUSTUP_INSTALL_CMD"
        read_input "Install Rust?"
        case "$REPLY" in
            [Nn]*)
                error "Rust is required to build mistral.rs from source"
                ;;
        esac
        install_rust
    fi

    # Run prereq installers outside any $() so xcodebuild stdout (asset paths with slashes) can't leak into the captured feature string.
    if [ "$SELECTED_BACKEND" = "metal" ]; then
        check_xcode_cli_tools
        check_metal_toolchain
    fi

    BUILD_FEATURES=$(build_features "$os")
}

maybe_install_ffmpeg() {
    [ "$INSTALL_VIDEO" = "1" ] || return 0
    [ "${MISTRALRS_INSTALL_IGNORE_FFMPEG:-}" != "1" ] || return 0
    if check_ffmpeg; then
        return
    fi
    if install_ffmpeg "$1" && check_ffmpeg; then
        success "FFmpeg installed successfully"
    else
        warn "FFmpeg could not be installed. You can add video support later."
    fi
}

# Render a path with $HOME collapsed to ~ for readability.
tildify() {
    case "$1" in
        "$HOME"/*) printf '~%s' "${1#$HOME}" ;;
        *) printf '%s' "$1" ;;
    esac
}

write_env_script() {
    printf '%s\n' \
        '#!/bin/sh' \
        '# mistral.rs shell setup' \
        'case ":${PATH}:" in' \
        '    *:"$HOME/.local/bin":*) ;;' \
        '    *) export PATH="$HOME/.local/bin:$PATH" ;;' \
        'esac' \
        > "$MISTRALRS_ENV"
}

append_source_line() {
    rc="$1"
    [ -n "$rc" ] || return 0
    mkdir -p "$(dirname "$rc")"
    touch "$rc"
    source_line='. "$HOME/.mistralrs/env"'
    if ! grep -Fqx "$source_line" "$rc"; then
        printf '\n%s\n' "$source_line" >> "$rc"
        PATH_RCS="${PATH_RCS}${PATH_RCS:+, }$(tildify "$rc")"
    fi
}

setup_shell_path() {
    PATH_RCS=""
    write_env_script
    append_source_line "$HOME/.profile"
    for rc in "$HOME/.bash_profile" "$HOME/.bash_login" "$HOME/.bashrc"; do
        [ -f "$rc" ] && append_source_line "$rc"
    done
    if command -v zsh >/dev/null 2>&1 || [ "${SHELL##*/}" = "zsh" ]; then
        append_source_line "${ZDOTDIR:-$HOME}/.zshenv"
    fi
    printf '%s' "$PATH_RCS"
}

warn_if_shadowed() {
    resolved=$(command -v mistralrs 2>/dev/null || true)
    [ -n "$resolved" ] || return 0
    case "$resolved" in
        "$BIN_DIR/mistralrs"|"$PREBUILT_DIR/mistralrs") ;;
        *)
            printf "${YELLOW}Note:${NC} another mistralrs appears earlier on PATH: %s\n" "$(tildify "$resolved")"
            printf "      The managed install is available at: %s\n\n" "$(tildify "$PREBUILT_DIR/mistralrs")"
            ;;
    esac
}

add_duplicate_install() {
    candidate="$1"
    [ -n "$candidate" ] && [ -f "$candidate" ] || return 0
    case "$candidate" in
        "$BIN_DIR/mistralrs"|"$PREBUILT_DIR/mistralrs") return 0 ;;
    esac
    case "
$DUPLICATE_INSTALLS
" in
        *"
$candidate
"*) return 0 ;;
    esac
    DUPLICATE_INSTALLS="${DUPLICATE_INSTALLS}${DUPLICATE_INSTALLS:+
}$candidate"
}

find_duplicate_installs() {
    DUPLICATE_INSTALLS=""
    add_duplicate_install "$CARGO_MISTRALRS"
    old_ifs=$IFS
    IFS=:
    for dir in $PATH; do
        [ -n "$dir" ] && add_duplicate_install "$dir/mistralrs"
    done
    IFS=$old_ifs
}

confirm_duplicate_replacement() {
    find_duplicate_installs
    [ -n "$DUPLICATE_INSTALLS" ] || return 0
    echo ""
    printf "${YELLOW}warning:${NC} Found duplicate mistralrs installs:\n"
    printf '%s\n' "$DUPLICATE_INSTALLS" | while IFS= read -r duplicate; do
        printf "  %s\n" "$(tildify "$duplicate")"
    done
    echo ""
    read_input "Replace duplicate installs?"
    case "$REPLY" in
        [Nn]*)
            error "duplicate mistralrs installs must be resolved before installing"
            ;;
    esac
    REPLACE_DUPLICATE_INSTALLS=1
}

remove_duplicate_installs() {
    [ "$REPLACE_DUPLICATE_INSTALLS" = "1" ] || return 0
    find_duplicate_installs
    [ -n "$DUPLICATE_INSTALLS" ] || return 0
    while IFS= read -r duplicate; do
        [ -n "$duplicate" ] || continue
        if rm -f "$duplicate"; then
            info "Removed duplicate install: $(tildify "$duplicate")"
        else
            error "failed to remove duplicate install: $(tildify "$duplicate")"
        fi
    done <<EOF
$DUPLICATE_INSTALLS
EOF
}

print_success() {
    ver=$("$PREBUILT_DIR/mistralrs" --version)
    printf "\n  ${GREEN}${BOLD}%s installed successfully.${NC}\n\n" "$ver"
    printf "  Command: %s\n\n" "$(tildify "$BIN_DIR/mistralrs")"
    case ":$PATH:" in
        *":$BIN_DIR:"*) ;;
        *)
            printf '  To use it in this terminal, run:\n\n'
            printf '    . "$HOME/.mistralrs/env"\n\n'
            ;;
    esac
    warn_if_shadowed
    printf '  Try browser chat:\n\n'
    printf '    mistralrs serve -m %s --quant %s\n\n' "$DEFAULT_MODEL" "$DEFAULT_QUANTIZATION"
    printf '  Then open %s\n\n' "$DEFAULT_UI_URL"
}

main() {
    load_options
    parse_args "$@"
    trap cleanup_installer 0
    trap 'exit 130' INT
    trap 'exit 143' TERM
    trap 'exit 129' HUP
    INSTALL_OS=$(detect_os)
    INSTALL_ARCH=$(uname -m)
    DETECTED_CUDA_CC=$(detect_cuda_compute_cap)
    print_banner
    guided_install
    if [ "$DRY_RUN" = "1" ]; then
        printf '  Preview complete. No files were changed.\n'
        return
    fi
    INSTALL_LOG=$(mktemp "${TMPDIR:-/tmp}/mistralrs-install.XXXXXX")
    begin_stage 1 "Prepare installation"
    configure_release
    confirm_duplicate_replacement
    if [ "$PLANNED_METHOD" = "prebuilt" ]; then
        stage_done "Ready to download"
        begin_stage 2 "Download binary"
        DOWNLOAD_STAGE=$(mktemp -d "$HOME/.mistralrs-stage.XXXXXX")
        if ! download_prebuilt "$PLANNED_ASSET"; then
            legacy_asset=$(detect_legacy_cuda_prebuilt_asset "$INSTALL_OS")
            if [ -z "$legacy_asset" ] || [ "$legacy_asset" = "$PLANNED_ASSET" ]; then
                error "Binary download failed. Check your connection and retry, or choose --from-source to build explicitly."
            fi
            stage_detail "Trying the older CUDA package name..."
            download_prebuilt "$legacy_asset" \
                || error "Binary download failed. Retry, or choose --from-source to build explicitly."
        fi
        PROGRESS_DOWNLOAD_PATH=""
        PROGRESS_DOWNLOAD_SIZE=""
        begin_stage 3 "Verify binary"
        INSTALL_CANDIDATE="$DOWNLOAD_STAGE/unpacked"
        mkdir -p "$INSTALL_CANDIDATE"
        run_logged "Unpack binary" tar xzf "$DOWNLOAD_STAGE/$asset" -C "$INSTALL_CANDIDATE" \
            || error "Could not unpack the download. Retry the installation. Existing installation kept."
        chmod +x "$INSTALL_CANDIDATE/mistralrs" 2>/dev/null || true
    else
        prepare_source_build "$INSTALL_OS"
        stage_done "Build tools ready"
        begin_stage 2 "Build from source"
        install_mistralrs "$BUILD_FEATURES"
        begin_stage 3 "Verify binary"
        stage_source_binary
    fi
    run_logged "Check that mistral.rs runs" "$INSTALL_CANDIDATE/mistralrs" --version \
        || error "The new binary could not run. Review the error above. Existing installation kept."
    begin_stage 4 "Finish setup"
    if ! activate_install "$INSTALL_CANDIDATE"; then
        error "Could not activate the new binary. Check the installation log before retrying."
    fi
    save_options
    remove_duplicate_installs
    maybe_install_ffmpeg "$INSTALL_OS"
    path_rcs=$(setup_shell_path)
    stage_done "Command and shell environment ready"
    print_success
}

# Parse the exit with main so a running install cannot read a modified script tail.
{
    main "$@"
    exit "$?"
}
