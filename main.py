#!/usr/bin/env python3
"""liveQwen — Real-time voice cloning with STT + Qwen3-TTS."""

import sys

from config import parse_args
from audio_io import list_devices, AudioOutput, VirtualMic
from stt import SpeechToText
from tts import VoiceCloner
from pipeline import LivePipeline, OfflinePipeline


def main():
    args = parse_args()

    # --list-devices: show devices and exit
    if args.list_devices:
        list_devices()
        return

    print("=" * 60)
    print("  liveQwen — Voice Clone Pipeline")
    print("=" * 60)
    print(f"  Mode:       {args.mode}")
    print(f"  STT model:  {args.stt_model} ({args.stt_device}, {args.stt_compute})")
    print(f"  TTS model:  {args.tts_model}")
    print(f"  Ref audio:  {args.ref_audio}")
    print(f"  Ref text:   {args.ref_text or '(auto-transcribe)'}")
    print(f"  Play:       {args.play}")
    print(f"  Save:       {args.save} → {args.output_dir}")
    if args.output_device:
        print(f"  Output dev: {args.output_device}")
    if args.mirror_device:
        print(f"  Mirror dev: {args.mirror_device}")
    if args.virtual_mic:
        print(f"  Virtual mic: enabled (auto)")
    if args.mode == "offline":
        print(f"  Input file: {args.input_file}")
    print("=" * 60)

    # 1. Load STT model (CPU by default to save GPU VRAM for TTS)
    stt = SpeechToText(
        model_size=args.stt_model,
        device=args.stt_device,
        compute_type=args.stt_compute,
    )

    # 2. Auto-transcribe reference audio if --ref-text not provided
    ref_text = args.ref_text
    if ref_text is None:
        print("[main] Auto-transcribing reference audio...")
        segments = stt.transcribe_file(args.ref_audio)
        ref_text = " ".join(seg["text"] for seg in segments)
        if not ref_text.strip():
            print("[main] WARNING: Could not transcribe reference audio. "
                  "Voice cloning quality may be reduced.")
            ref_text = ""
        else:
            print(f"[main] Reference transcript: \"{ref_text}\"")

    # 3. Load TTS model and prepare voice clone prompt
    tts = VoiceCloner(
        model_name=args.tts_model,
        use_flash_attn=not args.no_flash_attn,
    )
    tts.setup_voice(ref_audio=args.ref_audio, ref_text=ref_text)

    # 4. Set up audio output (with optional virtual mic)
    virtual_mic = None
    mirror_device = args.mirror_device

    if args.virtual_mic:
        virtual_mic = VirtualMic()
        vmic_index = virtual_mic.setup()
        if vmic_index is not None:
            mirror_device = str(vmic_index)
        else:
            print("[main] WARNING: Virtual mic setup failed. Continuing without it.")

    audio_out = AudioOutput(
        output_device=args.output_device,
        mirror_device=mirror_device,
        mirror_sink_name="LiveQwenVoiceClone" if args.virtual_mic else None,
    )

    # 5. Run pipeline
    if args.mode == "live":
        pipe = LivePipeline(
            stt=stt,
            tts=tts,
            audio_out=audio_out,
            input_device=args.input_device,
            output_dir=args.output_dir,
            do_play=args.play,
            do_save=args.save,
            silence_duration=args.silence,
            push_to_talk=args.push_to_talk,
        )
        pipe.start()
    else:
        pipe = OfflinePipeline(
            stt=stt,
            tts=tts,
            audio_out=audio_out,
            output_dir=args.output_dir,
        )
        pipe.process(
            input_file=args.input_file,
            do_play=args.play,
            do_save=args.save,
        )

    # Clean up virtual mic
    if virtual_mic is not None:
        virtual_mic.teardown()

    print("[main] Done.")


if __name__ == "__main__":
    main()
