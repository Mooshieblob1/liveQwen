#!/usr/bin/env bash
# ──────────────────────────────────────────────────────
#  liveQwen Launcher
#  Activates the conda env and walks you through options
# ──────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Disable kitty keyboard protocol if active.
# Terminals like kitty/foot/wezterm encode keys as CSI sequences (e.g. Enter → ESC[13u)
# which breaks bash's `read`. Pop all protocol levels to restore normal input.
printf '\x1b[>0u' 2>/dev/null || true   # push level 0 (disable enhancements)
printf '\x1b[<u'  2>/dev/null || true   # pop
printf '\x1b[<u'  2>/dev/null || true   # pop again in case shell pushed multiple

CONDA_ENV="liveqwen"

# ── Activate conda ──────────────────────────────────
if ! command -v conda &>/dev/null; then
    # Try sourcing conda init from common locations
    for f in "/opt/miniconda3/etc/profile.d/conda.sh" \
             "$HOME/miniforge3/etc/profile.d/conda.sh" \
             "$HOME/miniconda3/etc/profile.d/conda.sh" \
             "$HOME/anaconda3/etc/profile.d/conda.sh" \
             "/opt/conda/etc/profile.d/conda.sh" \
             "/opt/miniforge3/etc/profile.d/conda.sh"; do
        if [[ -f "$f" ]]; then
            source "$f"
            break
        fi
    done
fi

if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Please install Miniconda/Miniforge or activate it in your shell."
    exit 1
fi

echo "Activating conda environment: $CONDA_ENV"
conda activate "$CONDA_ENV" 2>/dev/null || {
    echo "ERROR: Could not activate '$CONDA_ENV'. Create it first:"
    echo "  conda create -n $CONDA_ENV python=3.12 -y"
    echo "  conda activate $CONDA_ENV && pip install -r requirements.txt"
    exit 1
}

# ── Helper ──────────────────────────────────────────
ask_yn() {
    local prompt="$1" default="${2:-n}"
    local yn
    if [[ "$default" == "y" ]]; then
        read -rp "$prompt [Y/n]: " yn
        yn="${yn:-y}"
    else
        read -rp "$prompt [y/N]: " yn
        yn="${yn:-n}"
    fi
    [[ "${yn,,}" == "y" ]]
}

# ── GUI or CLI? ────────────────────────────────────
echo ""
echo "Launch mode:"
echo "  1) GUI (graphical interface)"
echo "  2) CLI (terminal interactive)"
read -rp "Choose [1/2] (default: 1): " launch_mode
launch_mode="${launch_mode:-1}"

if [[ "$launch_mode" == "1" ]]; then
    echo ""
    echo "Launching GUI..."
    exec python gui.py
fi

# ── Discover reference audio ───────────────────────
mapfile -t WAVS < <(find voice_samples -maxdepth 1 -name '*.wav' -type f 2>/dev/null | sort)

if [[ ${#WAVS[@]} -eq 0 ]]; then
    echo "No reference WAV files found in voice_samples/"
    read -rp "Path to reference audio: " REF_AUDIO
elif [[ ${#WAVS[@]} -eq 1 ]]; then
    REF_AUDIO="${WAVS[0]}"
    echo "Using reference audio: $REF_AUDIO"
else
    echo ""
    echo "Available reference voices:"
    for i in "${!WAVS[@]}"; do
        echo "  $((i+1))) ${WAVS[$i]}"
    done
    read -rp "Choose [1-${#WAVS[@]}] (default: 1): " choice
    choice="${choice:-1}"
    REF_AUDIO="${WAVS[$((choice-1))]}"
    echo "Using: $REF_AUDIO"
fi

# ── Gather options ─────────────────────────────────
echo ""
echo "─── Options ───"

ARGS=(--ref-audio "$REF_AUDIO")

# Mode
if ask_yn "Use push-to-talk? (Enter to send, instead of auto-silence)" "y"; then
    ARGS+=(--push-to-talk)
fi

# Virtual mic
if ask_yn "Enable virtual mic? (for Discord/OBS)" "y"; then
    ARGS+=(--virtual-mic)
fi

# Play through speakers
if ask_yn "Play audio through speakers?" "y"; then
    ARGS+=(--play)
fi

# Save to files
if ask_yn "Save output to files?" "y"; then
    ARGS+=(--save)
fi

# Silence threshold (only relevant without push-to-talk)
if [[ ! " ${ARGS[*]} " =~ " --push-to-talk " ]]; then
    read -rp "Silence threshold in seconds (default: 2.5): " silence
    silence="${silence:-2.5}"
    ARGS+=(--silence "$silence")
fi

# Advanced options
if ask_yn "Configure advanced options?" "n"; then
    echo ""
    # STT device
    echo "  STT device options: cpu (saves VRAM), cuda (faster)"
    read -rp "  STT device (default: cpu): " stt_dev
    stt_dev="${stt_dev:-cpu}"
    ARGS+=(--stt-device "$stt_dev")

    if [[ "$stt_dev" == "cuda" ]]; then
        ARGS+=(--stt-compute "int8_float16")
    fi

    # STT model
    echo "  STT model sizes: tiny, base, small, medium, large-v3"
    read -rp "  STT model (default: medium): " stt_model
    if [[ -n "$stt_model" ]]; then
        ARGS+=(--stt-model "$stt_model")
    fi

    # Flash attention
    if ask_yn "  Disable flash attention for TTS?" "n"; then
        ARGS+=(--no-flash-attn)
    fi
fi

# ── Summary & launch ──────────────────────────────
echo ""
echo "═══════════════════════════════════════════"
echo "  Launching: python main.py ${ARGS[*]}"
echo "═══════════════════════════════════════════"
echo ""

exec python main.py "${ARGS[@]}"
