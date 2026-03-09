import numpy as np
import torch
from qwen_tts import Qwen3TTSModel

from config import DEFAULT_TTS_MODEL


class VoiceCloner:
    """Wraps Qwen3-TTS Base model for voice cloning."""

    def __init__(self, model_name: str = DEFAULT_TTS_MODEL, use_flash_attn: bool = True):
        attn_impl = "flash_attention_2" if use_flash_attn else "sdpa"
        print(f"[tts] Loading Qwen3-TTS model '{model_name}' (attn={attn_impl})...")

        try:
            self.model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            )
        except Exception:
            if use_flash_attn:
                print("[tts] flash_attention_2 failed, falling back to sdpa...")
                self.model = Qwen3TTSModel.from_pretrained(
                    model_name,
                    device_map="cuda:0",
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                )
            else:
                raise

        self._voice_clone_prompt = None
        print("[tts] Model loaded.")

    def setup_voice(self, ref_audio: str, ref_text: str):
        """Pre-build the voice clone prompt from reference audio.

        Args:
            ref_audio: Path to reference audio WAV file.
            ref_text: Transcript of the reference audio.
        """
        print(f"[tts] Building voice clone prompt from: {ref_audio}")
        self._voice_clone_prompt = self.model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        print("[tts] Voice clone prompt ready.")

    def clone_speech(self, text: str, language: str = "Auto") -> tuple[np.ndarray, int]:
        """Generate speech in the cloned voice.

        Args:
            text: Text to speak.
            language: Qwen3-TTS language name (e.g. "English", "Chinese", "Auto").

        Returns:
            (wav_array, sample_rate): The generated audio as float numpy array and its sample rate.
        """
        if self._voice_clone_prompt is None:
            raise RuntimeError("Voice not set up. Call setup_voice() first.")

        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=self._voice_clone_prompt,
        )

        return wavs[0], sr
