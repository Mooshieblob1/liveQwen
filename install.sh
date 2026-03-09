#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  liveQwen Installer
#  Sets up everything needed to run liveQwen from scratch:
#    - Python 3.12 via conda
#    - CUDA toolkit check
#    - PulseAudio utilities
#    - Python dependencies
#    - Flash Attention (optional)
# ──────────────────────────────────────────────────────────────
set -euo pipefail

CONDA_ENV="liveqwen"
PYTHON_VERSION="3.12"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colors ────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'  # No Color

info()  { echo -e "${BLUE}[info]${NC}  $*"; }
ok()    { echo -e "${GREEN}[  ok]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }

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

header() {
    echo ""
    echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  $*${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
    echo ""
}

# ── Detect package manager ────────────────────────────────────
detect_pkg_manager() {
    if command -v apt-get &>/dev/null; then
        echo "apt"
    elif command -v dnf &>/dev/null; then
        echo "dnf"
    elif command -v yum &>/dev/null; then
        echo "yum"
    elif command -v pacman &>/dev/null; then
        echo "pacman"
    elif command -v zypper &>/dev/null; then
        echo "zypper"
    else
        echo "unknown"
    fi
}

pkg_install() {
    local pkg_mgr="$1"
    shift
    case "$pkg_mgr" in
        apt)    sudo apt-get install -y "$@" ;;
        dnf)    sudo dnf install -y "$@" ;;
        yum)    sudo yum install -y "$@" ;;
        pacman) sudo pacman -S --noconfirm "$@" ;;
        zypper) sudo zypper install -y "$@" ;;
        *)      fail "No supported package manager found."; return 1 ;;
    esac
}

# ══════════════════════════════════════════════════════════════
header "liveQwen Installer"

echo "This script will set up everything needed to run liveQwen:"
echo "  1. Check/install conda (Miniconda)"
echo "  2. Create conda environment with Python $PYTHON_VERSION"
echo "  3. Check NVIDIA GPU & CUDA"
echo "  4. Install PulseAudio utilities"
echo "  5. Install Python dependencies"
echo "  6. Optionally install Flash Attention 2"
echo ""

if ! ask_yn "Continue with installation?" "y"; then
    echo "Aborted."
    exit 0
fi

PKG_MGR=$(detect_pkg_manager)
info "Detected package manager: $PKG_MGR"

# ── Step 1: Conda ────────────────────────────────────────────
header "Step 1/6: Conda"

CONDA_FOUND=false

# Try to source conda
for f in "/opt/miniconda3/etc/profile.d/conda.sh" \
         "$HOME/miniforge3/etc/profile.d/conda.sh" \
         "$HOME/miniconda3/etc/profile.d/conda.sh" \
         "$HOME/anaconda3/etc/profile.d/conda.sh" \
         "/opt/conda/etc/profile.d/conda.sh" \
         "/opt/miniforge3/etc/profile.d/conda.sh"; do
    if [[ -f "$f" ]]; then
        source "$f"
        CONDA_FOUND=true
        break
    fi
done

if command -v conda &>/dev/null; then
    CONDA_FOUND=true
fi

if $CONDA_FOUND; then
    ok "Conda found: $(conda --version 2>&1)"
else
    warn "Conda not found."
    if ask_yn "Install Miniconda now?" "y"; then
        info "Downloading Miniconda..."
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"
        
        if command -v curl &>/dev/null; then
            curl -fsSL "$MINICONDA_URL" -o "$MINICONDA_INSTALLER"
        elif command -v wget &>/dev/null; then
            wget -q "$MINICONDA_URL" -O "$MINICONDA_INSTALLER"
        else
            fail "Neither curl nor wget found. Install one first."
            exit 1
        fi

        info "Installing Miniconda to ~/miniconda3..."
        bash "$MINICONDA_INSTALLER" -b -p "$HOME/miniconda3"
        rm -f "$MINICONDA_INSTALLER"

        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda init bash &>/dev/null || true

        ok "Miniconda installed."
        echo ""
        warn "You may need to restart your shell after installation for conda"
        warn "to be available in future sessions."
    else
        fail "Conda is required. Install it manually from: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
fi

# ── Step 2: Conda Environment ────────────────────────────────
header "Step 2/6: Conda Environment"

if conda env list 2>/dev/null | grep -q "^${CONDA_ENV} "; then
    ok "Environment '$CONDA_ENV' already exists."
    if ask_yn "Recreate it from scratch?" "n"; then
        info "Removing existing environment..."
        conda env remove -n "$CONDA_ENV" -y
        info "Creating environment '$CONDA_ENV' with Python $PYTHON_VERSION..."
        conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -y
        ok "Environment recreated."
    fi
else
    info "Creating environment '$CONDA_ENV' with Python $PYTHON_VERSION..."
    conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -y
    ok "Environment created."
fi

info "Activating '$CONDA_ENV'..."
conda activate "$CONDA_ENV"

# Verify Python version
PY_VER=$(python --version 2>&1)
if [[ "$PY_VER" == *"$PYTHON_VERSION"* ]]; then
    ok "Python: $PY_VER"
else
    warn "Python version mismatch: got $PY_VER (expected $PYTHON_VERSION)"
    warn "This may still work, but $PYTHON_VERSION is recommended."
fi

# ── Step 3: NVIDIA GPU & CUDA ────────────────────────────────
header "Step 3/6: NVIDIA GPU & CUDA"

CUDA_OK=false

# Check for NVIDIA GPU
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "")
    if [[ -n "$GPU_INFO" ]]; then
        ok "NVIDIA GPU detected:"
        echo "$GPU_INFO" | while IFS= read -r line; do
            echo "      $line"
        done
        
        # Check VRAM
        VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        if [[ -n "$VRAM_MB" ]] && (( VRAM_MB >= 8000 )); then
            ok "VRAM: ${VRAM_MB} MB (≥ 8 GB — sufficient)"
        elif [[ -n "$VRAM_MB" ]]; then
            warn "VRAM: ${VRAM_MB} MB — less than 8 GB. You may need the smaller TTS model."
        fi
    else
        warn "nvidia-smi found but no GPU detected."
    fi
else
    fail "nvidia-smi not found. NVIDIA drivers may not be installed."
    echo ""
    echo "  liveQwen requires an NVIDIA GPU with CUDA support."
    echo "  Install NVIDIA drivers for your distribution:"
    case "$PKG_MGR" in
        apt)    echo "    sudo apt install nvidia-driver-550" ;;
        dnf)    echo "    sudo dnf install akmod-nvidia" ;;
        pacman) echo "    sudo pacman -S nvidia" ;;
        *)      echo "    See: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/" ;;
    esac
    echo ""
    if ! ask_yn "Continue anyway?" "n"; then
        exit 1
    fi
fi

# Check CUDA toolkit via PyTorch (will be verified after pip install)
info "CUDA runtime will be verified after PyTorch installation."

# ── Step 4: PulseAudio ───────────────────────────────────────
header "Step 4/6: PulseAudio Utilities"

if command -v pactl &>/dev/null; then
    ok "pactl found: $(pactl --version 2>/dev/null | head -1)"
else
    warn "pactl not found (needed for virtual microphone support)."
    if ask_yn "Install PulseAudio utilities now?" "y"; then
        case "$PKG_MGR" in
            apt)    pkg_install "$PKG_MGR" pulseaudio-utils ;;
            dnf)    pkg_install "$PKG_MGR" pulseaudio-utils ;;
            yum)    pkg_install "$PKG_MGR" pulseaudio-utils ;;
            pacman) pkg_install "$PKG_MGR" libpulse ;;
            zypper) pkg_install "$PKG_MGR" pulseaudio-utils ;;
            *)      fail "Unknown package manager. Install PulseAudio utilities manually." ;;
        esac

        if command -v pactl &>/dev/null; then
            ok "pactl installed successfully."
        else
            warn "pactl still not found. Virtual mic feature won't work."
        fi
    else
        warn "Skipping. Virtual mic (--virtual-mic) won't be available."
    fi
fi

# Check if PulseAudio/PipeWire is actually running
if pactl info &>/dev/null 2>&1; then
    SERVER_NAME=$(pactl info 2>/dev/null | grep "Server Name" | cut -d: -f2- | xargs)
    ok "Audio server running: $SERVER_NAME"
else
    warn "PulseAudio/PipeWire doesn't seem to be running."
    warn "Virtual mic features may not work until it's started."
fi

# Also check for paplay (used for mirror output)
if command -v paplay &>/dev/null; then
    ok "paplay found (used for virtual mic audio routing)."
else
    warn "paplay not found. It's usually included with pulseaudio-utils."
fi

# ── Step 5: Python Dependencies ──────────────────────────────
header "Step 5/6: Python Dependencies"

if [[ ! -f "$SCRIPT_DIR/requirements.txt" ]]; then
    fail "requirements.txt not found in $SCRIPT_DIR"
    exit 1
fi

info "Installing Python packages..."
pip install -r "$SCRIPT_DIR/requirements.txt"
echo ""

# Verify key imports
info "Verifying installations..."

VERIFY_FAILED=false

# PyTorch + CUDA
TORCH_INFO=$(python -c "
import torch
cuda = torch.cuda.is_available()
ver = torch.version.cuda if cuda else 'N/A'
dev = torch.cuda.get_device_name(0) if cuda else 'N/A'
print(f'PyTorch {torch.__version__} | CUDA available: {cuda} | CUDA version: {ver} | Device: {dev}')
" 2>&1) || TORCH_INFO="FAILED"

if [[ "$TORCH_INFO" == "FAILED" ]]; then
    fail "PyTorch import failed. Try: pip install torch --index-url https://download.pytorch.org/whl/cu121"
    VERIFY_FAILED=true
elif [[ "$TORCH_INFO" == *"CUDA available: False"* ]]; then
    warn "$TORCH_INFO"
    echo ""
    warn "PyTorch was installed without CUDA support."
    echo "  To fix, reinstall PyTorch with CUDA:"
    echo "    pip install torch --index-url https://download.pytorch.org/whl/cu121"
    echo "  (Replace cu121 with your CUDA version: cu118, cu121, cu124)"
    echo ""
    if ask_yn "Install PyTorch with CUDA 12.1 now?" "y"; then
        pip install torch --index-url https://download.pytorch.org/whl/cu121
        # Re-check
        TORCH_INFO=$(python -c "
import torch
cuda = torch.cuda.is_available()
ver = torch.version.cuda if cuda else 'N/A'
dev = torch.cuda.get_device_name(0) if cuda else 'N/A'
print(f'PyTorch {torch.__version__} | CUDA available: {cuda} | CUDA version: {ver} | Device: {dev}')
" 2>&1) || true
        if [[ "$TORCH_INFO" == *"CUDA available: True"* ]]; then
            ok "$TORCH_INFO"
        else
            warn "CUDA still not available: $TORCH_INFO"
            warn "You may need a different CUDA version. Check: nvidia-smi"
        fi
    fi
else
    ok "$TORCH_INFO"
fi

# faster-whisper
FW_VER=$(python -c "import faster_whisper; print(f'faster-whisper {faster_whisper.__version__}')" 2>&1) || FW_VER="FAILED"
if [[ "$FW_VER" == "FAILED" ]]; then
    fail "faster-whisper import failed"
    VERIFY_FAILED=true
else
    ok "$FW_VER"
fi

# qwen-tts
QT_VER=$(python -c "from qwen_tts import QwenTTS; print('qwen-tts OK')" 2>&1) || QT_VER="FAILED"
if [[ "$QT_VER" == "FAILED" ]]; then
    fail "qwen-tts import failed"
    VERIFY_FAILED=true
else
    ok "$QT_VER"
fi

# sounddevice
SD_VER=$(python -c "import sounddevice; print(f'sounddevice {sounddevice.__version__}')" 2>&1) || SD_VER="FAILED"
if [[ "$SD_VER" == "FAILED" ]]; then
    fail "sounddevice import failed. You may need: sudo $( [[ $PKG_MGR == pacman ]] && echo 'pacman -S portaudio' || echo 'apt install libportaudio2' )"
    VERIFY_FAILED=true
else
    ok "$SD_VER"
fi

if $VERIFY_FAILED; then
    echo ""
    warn "Some packages failed to import. Fix the issues above before running liveQwen."
else
    echo ""
    ok "All core packages verified."
fi

# ── Step 6: Flash Attention (Optional) ───────────────────────
header "Step 6/6: Flash Attention 2 (Optional)"

echo "Flash Attention 2 can reduce GPU memory usage and speed up TTS inference."
echo "It requires a compatible GPU (Ampere or newer) and takes time to compile."
echo ""

FA_INSTALLED=$(python -c "import flash_attn; print('yes')" 2>/dev/null || echo "no")

if [[ "$FA_INSTALLED" == "yes" ]]; then
    FA_VER=$(python -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null)
    ok "Flash Attention already installed (v$FA_VER)"
else
    if ask_yn "Install Flash Attention 2? (may take 10+ minutes to compile)" "n"; then
        info "Installing flash-attn..."
        
        # Determine MAX_JOBS based on available RAM
        TOTAL_RAM_GB=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')
        if [[ -n "$TOTAL_RAM_GB" ]] && (( TOTAL_RAM_GB < 96 )); then
            info "System has ${TOTAL_RAM_GB}GB RAM — limiting compile jobs to 4"
            MAX_JOBS=4 pip install -U flash-attn --no-build-isolation || {
                warn "Flash Attention installation failed. This is optional — liveQwen will use sdpa fallback."
            }
        else
            pip install -U flash-attn --no-build-isolation || {
                warn "Flash Attention installation failed. This is optional — liveQwen will use sdpa fallback."
            }
        fi

        # Verify
        FA_INSTALLED=$(python -c "import flash_attn; print('yes')" 2>/dev/null || echo "no")
        if [[ "$FA_INSTALLED" == "yes" ]]; then
            ok "Flash Attention installed successfully."
        else
            warn "Flash Attention not available. liveQwen will fall back to sdpa (still works fine)."
        fi
    else
        info "Skipping. liveQwen will use sdpa attention (no flash-attn needed)."
    fi
fi

# ── Summary ──────────────────────────────────────────────────
header "Installation Complete"

echo "Summary:"
echo "  Conda env:       $CONDA_ENV (Python $PYTHON_VERSION)"
echo "  PyTorch:         $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'ERROR')"
echo "  CUDA available:  $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'ERROR')"
echo "  faster-whisper:  $(python -c 'import faster_whisper; print(faster_whisper.__version__)' 2>/dev/null || echo 'ERROR')"
echo "  Flash Attention: $( [[ "$FA_INSTALLED" == "yes" ]] && echo "installed" || echo "not installed (optional)" )"
echo "  pactl:           $(command -v pactl &>/dev/null && echo "available" || echo "not found")"
echo "  paplay:          $(command -v paplay &>/dev/null && echo "available" || echo "not found")"
echo ""
echo "To run liveQwen:"
echo ""
echo -e "  ${GREEN}./launch.sh${NC}"
echo ""
echo "Or manually:"
echo ""
echo -e "  ${GREEN}conda activate $CONDA_ENV${NC}"
echo -e "  ${GREEN}python main.py --ref-audio voice_samples/your_voice.wav --virtual-mic --push-to-talk --play${NC}"
echo ""
