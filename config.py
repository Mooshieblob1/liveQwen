import argparse
import os

# Audio defaults
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 512            # ~32ms at 16kHz
SILENCE_THRESHOLD_S = 2.0   # seconds of silence to end an utterance

# Model defaults
DEFAULT_STT_MODEL = "medium"
DEFAULT_STT_DEVICE = "cpu"
DEFAULT_STT_COMPUTE = "int8"
DEFAULT_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_TTS_MODEL_SMALL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

# Output defaults
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Whisper language code → Qwen3-TTS language name
LANGUAGE_MAP = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}


def map_language(whisper_code: str) -> str:
    """Map a Whisper language code to a Qwen3-TTS language name."""
    if whisper_code is None:
        return "Auto"
    return LANGUAGE_MAP.get(whisper_code, "Auto")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="liveQwen — Real-time voice cloning with STT + Qwen3-TTS",
    )

    p.add_argument(
        "--mode",
        choices=["live", "offline"],
        default="live",
        help="Pipeline mode: 'live' for mic input, 'offline' for file input (default: live)",
    )

    # Reference voice
    p.add_argument(
        "--ref-audio",
        required=True,
        help="Path to reference voice WAV file for voice cloning",
    )
    p.add_argument(
        "--ref-text",
        default=None,
        help="Transcript of the reference audio. If omitted, auto-transcribed via STT.",
    )

    # Offline mode input
    p.add_argument(
        "--input-file",
        default=None,
        help="Input audio file path (required for offline mode)",
    )

    # Output
    p.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for saved WAV output (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--output-device",
        default=None,
        help="Output audio device name or index (for virtual device routing)",
    )
    p.add_argument(
        "--mirror-device",
        default=None,
        help="Second output device to mirror audio to (e.g. virtual mic name or index)",
    )
    p.add_argument(
        "--virtual-mic",
        action="store_true",
        help="Auto-create a PulseAudio virtual mic and mirror audio to it",
    )
    p.add_argument(
        "--play",
        action="store_true",
        help="Play cloned audio through speakers",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="Save cloned audio to WAV files in --output-dir",
    )

    # Models
    p.add_argument(
        "--stt-model",
        default=DEFAULT_STT_MODEL,
        help=f"Faster-whisper model size (default: {DEFAULT_STT_MODEL})",
    )
    p.add_argument(
        "--stt-device",
        default=DEFAULT_STT_DEVICE,
        choices=["cpu", "cuda"],
        help=f"Device for STT model — 'cpu' recommended to save GPU VRAM for TTS (default: {DEFAULT_STT_DEVICE})",
    )
    p.add_argument(
        "--stt-compute",
        default=DEFAULT_STT_COMPUTE,
        help=f"Compute type for STT model, e.g. int8, float16, float32 (default: {DEFAULT_STT_COMPUTE})",
    )
    p.add_argument(
        "--tts-model",
        default=DEFAULT_TTS_MODEL,
        help=f"Qwen3-TTS model name or path (default: {DEFAULT_TTS_MODEL})",
    )
    p.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Disable flash attention for TTS model",
    )

    # Audio tuning
    p.add_argument(
        "--silence",
        type=float,
        default=SILENCE_THRESHOLD_S,
        help=f"Seconds of silence before an utterance is considered complete (default: {SILENCE_THRESHOLD_S})",
    )
    p.add_argument(
        "--push-to-talk",
        action="store_true",
        help="Use push-to-talk mode: press Enter when done speaking instead of auto-detecting silence",
    )

    # Audio device selection
    p.add_argument(
        "--input-device",
        default=None,
        help="Input audio device name or index for microphone",
    )

    # Misc
    p.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )

    args = p.parse_args(argv)

    # Validation
    if args.mode == "offline" and args.input_file is None:
        p.error("--input-file is required for offline mode")

    if not os.path.isfile(args.ref_audio):
        p.error(f"Reference audio file not found: {args.ref_audio}")

    if args.mode == "offline" and args.input_file and not os.path.isfile(args.input_file):
        p.error(f"Input file not found: {args.input_file}")

    # Default: if neither --play nor --save, enable both
    if not args.play and not args.save:
        args.play = True
        args.save = True

    os.makedirs(args.output_dir, exist_ok=True)

    return args
