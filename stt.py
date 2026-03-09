import numpy as np
from faster_whisper import WhisperModel

from config import SAMPLE_RATE, map_language


class SpeechToText:
    """Wraps faster-whisper for speech-to-text transcription."""

    def __init__(self, model_size: str = "medium", device: str = "cuda",
                 compute_type: str = "float16"):
        print(f"[stt] Loading faster-whisper model '{model_size}' on {device} ({compute_type})...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("[stt] Model loaded.")

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> tuple[str, str]:
        """Transcribe audio to text.

        Args:
            audio: float32 numpy array of audio samples.
            sample_rate: Sample rate of the audio (must be 16000 for Whisper).

        Returns:
            (text, qwen_language): The transcribed text and the mapped Qwen3-TTS language name.
        """
        # faster-whisper expects float32 numpy array at 16kHz
        if sample_rate != 16000:
            raise ValueError(f"Whisper requires 16kHz audio, got {sample_rate}Hz")

        segments, info = self.model.transcribe(
            audio,
            beam_size=5,
            vad_filter=True,
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        full_text = " ".join(text_parts)
        qwen_lang = map_language(info.language)

        return full_text, qwen_lang

    def transcribe_file(self, path: str) -> list[dict]:
        """Transcribe an audio file and return segments with timestamps.

        Returns:
            List of dicts with keys: 'start', 'end', 'text', 'language'.
        """
        segments, info = self.model.transcribe(
            path,
            beam_size=5,
            vad_filter=True,
        )

        qwen_lang = map_language(info.language)
        results = []
        for seg in segments:
            results.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "language": qwen_lang,
            })

        return results
