# liveQwen

A real-time voice cloning pipeline that captures your speech, transcribes it, and re-synthesizes it in a cloned voice using **Qwen3-TTS**. Supports live microphone input, offline file processing, and output to speakers, WAV files, or a virtual audio device (for Discord, OBS, etc.).

## How It Works

```
Microphone → [faster-whisper STT] → Text → [Qwen3-TTS Voice Clone] → Cloned Audio → Speakers / File / Virtual Mic
```

1. **Speech-to-Text**: Your speech is transcribed in real-time using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with automatic language detection.
2. **Voice Cloning**: The transcribed text is fed to [Qwen3-TTS-12Hz-Base](https://github.com/QwenLM/Qwen3-TTS) which generates speech in the voice of your provided reference audio sample.
3. **Output**: The cloned audio is played through speakers, saved to files, and/or routed to a virtual audio device.

> **Note**: This is an STT → TTS pipeline. The original intonation/prosody of your speech is **not** preserved — Qwen3-TTS generates contextually appropriate prosody from the text. The voice **timbre** matches your reference audio.

## Requirements

- **GPU**: NVIDIA GPU with ≥ 12GB VRAM (tested with 12GB)
- **OS**: Linux (PulseAudio for virtual audio device)
- **Python**: 3.12 recommended
- **CUDA**: Compatible with PyTorch 2.1+

## Setup

### 1. Create Environment

```bash
conda create -n liveqwen python=3.12 -y
conda activate liveqwen
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Install Flash Attention 2

Reduces GPU memory usage. Skip if your GPU doesn't support it or if you have < 96GB RAM:

```bash
# If less than 96GB RAM:
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation

# Otherwise:
pip install -U flash-attn --no-build-isolation
```

### 4. Prepare a Reference Voice Sample

Place a short (~5–15 second) WAV file of the target voice in `voice_samples/`. Clear speech with minimal background noise works best.

## Quick Start

The easiest way to run liveQwen is via the interactive launcher:

```bash
./launch.sh
```

This will:
- Automatically activate the `liveqwen` conda environment
- Detect reference voices in `voice_samples/`
- Walk you through options interactively (push-to-talk, virtual mic, etc.)
- Launch the pipeline with the chosen settings

## Usage

### Interactive Launcher (Recommended)

```bash
./launch.sh
```

The launcher prompts for all options with sensible defaults:
- **Push-to-talk** (default: yes) — press Enter to send each utterance
- **Virtual mic** (default: yes) — auto-creates a PulseAudio virtual microphone for Discord/OBS
- **Play through speakers** (default: yes)
- **Save to files** (default: yes)
- **Advanced options** — STT device (cpu/cuda), model size, flash attention

### Direct CLI Usage

Make sure to activate the conda env first:

```bash
conda activate liveqwen
```

#### Live Mode (Real-Time Mic Input)

```bash
# Basic — push-to-talk with virtual mic and speaker output
python main.py --ref-audio voice_samples/your_voice.wav \
               --push-to-talk --virtual-mic --play

# Auto-detect silence instead of push-to-talk (2.5s silence threshold)
python main.py --ref-audio voice_samples/your_voice.wav \
               --virtual-mic --play --silence 2.5

# With manual reference transcript (can improve clone quality)
python main.py --ref-audio voice_samples/your_voice.wav \
               --ref-text "The exact words spoken in the reference audio." \
               --push-to-talk --play

# Save utterances to files (no speaker playback)
python main.py --ref-audio voice_samples/your_voice.wav \
               --push-to-talk --save

# Faster STT on GPU (uses ~0.8 GB VRAM)
python main.py --ref-audio voice_samples/your_voice.wav \
               --push-to-talk --play --stt-device cuda --stt-compute int8_float16

# Smaller/faster STT model
python main.py --ref-audio voice_samples/your_voice.wav \
               --push-to-talk --play --stt-model small
```

#### Offline Mode (Process a File)

```bash
python main.py --mode offline \
               --ref-audio voice_samples/your_voice.wav \
               --input-file path/to/recording.wav
```

#### List Audio Devices

```bash
python main.py --ref-audio voice_samples/your_voice.wav --list-devices
```

### All CLI Options

| Flag | Default | Description |
|---|---|---|
| `--ref-audio PATH` | *(required)* | Reference voice WAV file for cloning |
| `--ref-text TEXT` | *(auto)* | Transcript of reference audio (auto-transcribed if omitted) |
| `--mode live\|offline` | `live` | Pipeline mode |
| `--push-to-talk` | off | Press Enter to send each utterance (instead of auto-silence) |
| `--silence SECONDS` | `2.0` | Seconds of silence to end an utterance (VAD mode) |
| `--virtual-mic` | off | Auto-create a PulseAudio virtual mic for Discord/OBS |
| `--mirror-device NAME\|IDX` | none | Second output device to mirror audio to |
| `--play` | off | Play cloned audio through speakers |
| `--save` | off | Save cloned audio to WAV files |
| `--output-dir PATH` | `./output/` | Directory for saved WAV output |
| `--output-device NAME\|IDX` | default | Primary output audio device |
| `--input-device NAME\|IDX` | default | Input microphone device |
| `--stt-model SIZE` | `medium` | Whisper model: tiny/base/small/medium/large-v3 |
| `--stt-device cpu\|cuda` | `cpu` | Device for STT model |
| `--stt-compute TYPE` | `int8` | Compute type: int8, int8_float16, float16, float32 |
| `--tts-model MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Qwen3-TTS model name or path |
| `--no-flash-attn` | off | Disable flash attention for TTS |
| `--input-file PATH` | none | Input audio file (required for offline mode) |
| `--list-devices` | off | List audio devices and exit |

## Virtual Audio Device (Discord / OBS)

### Automatic (Recommended)

Use the `--virtual-mic` flag and liveQwen handles everything:

```bash
python main.py --ref-audio voice_samples/your_voice.wav --virtual-mic --play
```

This creates:
1. A **PulseAudio null-sink** — the audio destination for cloned voice output
2. A **remapped source** — appears as **"liveQwen Microphone"** in app input device lists

In Discord/OBS, select **"liveQwen Microphone"** as your input device. Both are cleaned up automatically when the program exits.

If `pactl` (PulseAudio utilities) isn't installed, the launcher will attempt to install it automatically via your system package manager (apt, dnf, yum, pacman, or zypper).

### Manual Setup

If you prefer manual control:

```bash
# Create a virtual audio sink
pactl load-module module-null-sink sink_name=VoiceClone \
      sink_properties=device.description="VoiceClone"

# Create a remapped source so it appears as a microphone
pactl load-module module-remap-source source_name=VoiceCloneMic \
      master=VoiceClone.monitor \
      source_properties=device.description="VoiceClone Mic"

# Use it
python main.py --ref-audio voice_samples/your_voice.wav \
               --mirror-device "VoiceClone" --play

# Clean up when done
pactl unload-module module-remap-source
pactl unload-module module-null-sink
```

## Voice Activity Detection

liveQwen uses **Silero VAD** (neural network) for detecting when you stop speaking. Two input modes are available:

- **Push-to-talk** (`--push-to-talk`): Records until you press Enter. Best for reliable results — no accidental cutoffs.
- **Auto-detect silence** (default): Ends the utterance after `--silence` seconds (default: 2.0s) of silence. Good for hands-free use.

## Session Recording

In live mode, all utterances from a session are automatically combined into a single WAV file (`output/session_YYYYMMDD_HHMMSS.wav`) when you exit with Ctrl+C. The system keeps the last 10 sessions and cleans up older ones.

## Model Options

| TTS Model | VRAM | Quality | Speed |
|---|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (default) | ~3.4 GB | Higher | Slower |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | ~1.2 GB | Good | Faster |

| STT Model | VRAM (GPU) / RAM (CPU) | Quality | Speed |
|---|---|---|---|
| `tiny` | ~0.1 GB | Low | Fastest |
| `base` | ~0.2 GB | Fair | Fast |
| `small` | ~0.5 GB | Good | Medium |
| `medium` (default) | ~0.8 GB | High | Slower |
| `large-v3` | ~1.5 GB | Highest | Slowest |

### Memory Tips

- **Default config** (STT on CPU + TTS on GPU): ~3.4 GB VRAM
- **STT on GPU** (`--stt-device cuda --stt-compute int8_float16`): adds ~0.5–1.5 GB VRAM depending on model
- **Flash Attention**: reduces TTS VRAM usage; install if you have the build resources
- **Smaller TTS model** (`--tts-model Qwen/Qwen3-TTS-12Hz-0.6B-Base`): ~1.2 GB VRAM

## Project Structure

```
liveQwen/
├── launch.sh          — Interactive launcher (activates conda, prompts for options)
├── main.py            — Entry point
├── config.py          — CLI args and constants
├── audio_io.py        — Microphone capture, VAD, playback, virtual mic management
├── stt.py             — Speech-to-Text (faster-whisper)
├── tts.py             — Qwen3-TTS voice cloning
├── pipeline.py        — Live and offline pipeline orchestration (3-thread architecture)
├── requirements.txt   — Python dependencies
├── voice_samples/     — Place reference voice WAV files here
└── output/            — Generated output WAV files and session recordings
```

## Architecture

```
┌─────────────┐    ┌───────────────┐    ┌───────────────┐
│ Capture      │    │ Process       │    │ Output        │
│ Thread       │───>│ Thread        │───>│ Thread        │
│              │    │               │    │               │
│ Mic → VAD    │    │ STT → TTS    │    │ Speakers      │
│ or PTT       │    │               │    │ + File save   │
│              │    │               │    │ + Virtual mic │
└─────────────┘    └───────────────┘    └───────────────┘
     queue.Queue        queue.Queue
```

Three daemon threads connected by queues. The capture thread records audio and detects utterance boundaries (via Silero VAD or push-to-talk). The process thread transcribes and synthesizes. The output thread plays and saves.

## Supported Languages

Auto-detected by Whisper, supported by Qwen3-TTS:

Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian.
