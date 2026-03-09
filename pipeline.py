import glob
import os
import time
import queue
import threading
from datetime import datetime
import numpy as np
import soundfile as sf

from stt import SpeechToText
from tts import VoiceCloner
from audio_io import MicrophoneStream, UtteranceDetector, PushToTalkDetector, AudioOutput
from config import SAMPLE_RATE


class LivePipeline:
    """Real-time pipeline: mic → STT → TTS voice clone → output.

    Three threads:
      1. Capture thread: reads mic chunks, detects utterances via VAD
      2. Processing thread: STT + TTS (sequential on GPU)
      3. Output thread: plays/saves generated audio
    """

    def __init__(self, stt: SpeechToText, tts: VoiceCloner, audio_out: AudioOutput,
                 input_device=None, output_dir=None, do_play=True, do_save=True,
                 silence_duration=2.0, push_to_talk=False):
        self.stt = stt
        self.tts = tts
        self.audio_out = audio_out
        self.output_dir = output_dir
        self.do_play = do_play
        self.do_save = do_save
        self.push_to_talk = push_to_talk

        self._mic = MicrophoneStream(device=input_device)
        if push_to_talk:
            self._detector = PushToTalkDetector(sample_rate=SAMPLE_RATE)
        else:
            self._detector = UtteranceDetector(sample_rate=SAMPLE_RATE, silence_duration=silence_duration)

        self._utterance_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._output_queue: queue.Queue[tuple[np.ndarray, int, str] | None] = queue.Queue()

        self._running = False
        self._utterance_count = 0
        self._session_wavs: list[tuple[np.ndarray, int]] = []  # (wav, sr) for session combine
        self._session_lock = threading.Lock()

        # Clean up leftover utterance files from previous sessions
        self._cleanup_utterance_files()

    def _cleanup_utterance_files(self):
        """Remove any utterance_*.wav files left over from previous sessions."""
        if not self.output_dir:
            return
        pattern = os.path.join(self.output_dir, "utterance_*.wav")
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except OSError:
                pass

    def start(self):
        """Start the live pipeline."""
        self._running = True
        self._mic.start()
        if self.push_to_talk:
            self._detector.start()

        threads = [
            threading.Thread(target=self._capture_loop, name="capture", daemon=True),
            threading.Thread(target=self._process_loop, name="process", daemon=True),
            threading.Thread(target=self._output_loop, name="output", daemon=True),
        ]

        for t in threads:
            t.start()

        if self.push_to_talk:
            print("\n[pipeline] Push-to-talk mode. Press Enter when done speaking. Ctrl+C to quit.\n")
        else:
            print("\n[pipeline] Live mode active. Speak into your microphone. Press Ctrl+C to stop.\n")

        self._signal_listening()

        try:
            # Block main thread until interrupted
            while self._running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass

        self.stop()

        # Wait for threads to finish
        for t in threads:
            t.join(timeout=5)

        # Combine all utterances into a session file
        if self.do_save:
            self._save_session()

    def stop(self):
        """Stop the live pipeline gracefully."""
        if not self._running:
            return
        print("\n[pipeline] Stopping...")
        self._running = False
        self._mic.stop()
        if self.push_to_talk:
            self._detector.stop()
        # Signal processing thread
        self._utterance_queue.put(None)
        # Signal output thread
        self._output_queue.put(None)

    def _capture_loop(self):
        """Read mic chunks and detect utterance boundaries."""
        while self._running:
            chunk = self._mic.read(timeout=1)
            if chunk is None:
                break

            utterance = self._detector.feed(chunk)
            if utterance is not None:
                self._utterance_queue.put(utterance)

        # Flush any remaining audio
        final = self._detector.flush()
        if final is not None:
            self._utterance_queue.put(final)

    def _process_loop(self):
        """Take utterances, run STT then TTS, push output audio."""
        while self._running:
            try:
                utterance = self._utterance_queue.get(timeout=1)
            except queue.Empty:
                continue

            if utterance is None:
                break

            duration = len(utterance) / SAMPLE_RATE
            print(f"\r[pipeline] Utterance detected ({duration:.1f}s), processing...      ")

            t0 = time.time()

            # STT
            text, language = self.stt.transcribe(utterance, SAMPLE_RATE)
            t_stt = time.time() - t0

            if not text.strip():
                print("[pipeline] (empty transcription, skipping)")
                self._signal_listening()
                continue

            print(f"[pipeline] [{language}] \"{text}\"  (STT: {t_stt:.2f}s)")

            # TTS
            t1 = time.time()
            wav, sr = self.tts.clone_speech(text, language)
            t_tts = time.time() - t1

            print(f"[pipeline] TTS done ({t_tts:.2f}s, total: {t_stt + t_tts:.2f}s)")

            self._utterance_count += 1
            label = f"utterance_{self._utterance_count:04d}"

            self._output_queue.put((wav, sr, label))

    def _signal_listening(self):
        """Print a listening indicator and play a ready beep."""
        print("\n🎤 [LISTENING] Speak now...\n")
        if self.do_play:
            self.audio_out.play_ready_beep()

    def _output_loop(self):
        """Play and/or save output audio."""
        while self._running:
            try:
                item = self._output_queue.get(timeout=1)
            except queue.Empty:
                continue

            if item is None:
                break

            wav, sr, label = item

            # Collect for session combine
            with self._session_lock:
                self._session_wavs.append((wav, sr))

            if self.do_save and self.output_dir:
                path = os.path.join(self.output_dir, f"{label}.wav")
                self.audio_out.save(wav, sr, path)

            if self.do_play:
                self.audio_out.play_and_wait(wav, sr)

            # Signal ready for next utterance
            self._signal_listening()

    def _save_session(self):
        """Combine all utterances into a single session WAV file."""
        with self._session_lock:
            wavs = list(self._session_wavs)

        if not wavs or not self.output_dir:
            return

        sr = wavs[0][1]
        silence_gap = np.zeros(int(0.5 * sr), dtype=np.float32)

        parts = []
        for i, (wav, _) in enumerate(wavs):
            parts.append(wav)
            if i < len(wavs) - 1:
                parts.append(silence_gap)

        combined = np.concatenate(parts)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_path = os.path.join(self.output_dir, f"session_{timestamp}.wav")
        self.audio_out.save(combined, sr, session_path)

        # Enforce max 10 sessions
        self._cleanup_old_sessions(max_sessions=10)

    def _cleanup_old_sessions(self, max_sessions=10):
        """Keep only the most recent session files, delete the rest."""
        if not self.output_dir:
            return

        pattern = os.path.join(self.output_dir, "session_*.wav")
        session_files = sorted(glob.glob(pattern))

        if len(session_files) > max_sessions:
            for old_file in session_files[:-max_sessions]:
                try:
                    os.remove(old_file)
                    print(f"[pipeline] Removed old session: {os.path.basename(old_file)}")
                except OSError:
                    pass


class OfflinePipeline:
    """Offline pipeline: audio file → STT → TTS voice clone → output file."""

    def __init__(self, stt: SpeechToText, tts: VoiceCloner, audio_out: AudioOutput,
                 output_dir=None):
        self.stt = stt
        self.tts = tts
        self.audio_out = audio_out
        self.output_dir = output_dir

    def process(self, input_file: str, do_play=True, do_save=True):
        """Process an input audio file through the full pipeline."""
        print(f"[pipeline] Offline mode — processing: {input_file}")

        t0 = time.time()

        # Transcribe entire file into segments
        segments = self.stt.transcribe_file(input_file)
        t_stt = time.time() - t0

        if not segments:
            print("[pipeline] No speech detected in input file.")
            return

        print(f"[pipeline] Transcribed {len(segments)} segment(s) in {t_stt:.2f}s:")
        for i, seg in enumerate(segments):
            print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s] [{seg['language']}] \"{seg['text']}\"")

        # Clone each segment
        all_wavs = []
        total_tts = 0.0

        for i, seg in enumerate(segments):
            text = seg["text"]
            language = seg["language"]

            if not text.strip():
                continue

            print(f"[pipeline] Cloning segment {i + 1}/{len(segments)}...")
            t1 = time.time()
            wav, sr = self.tts.clone_speech(text, language)
            dt = time.time() - t1
            total_tts += dt
            print(f"[pipeline] Segment {i + 1} done ({dt:.2f}s)")

            all_wavs.append(wav)

        if not all_wavs:
            print("[pipeline] No audio generated.")
            return

        # Concatenate with small silence gaps between segments
        silence_gap = np.zeros(int(0.3 * sr), dtype=all_wavs[0].dtype)
        combined_parts = []
        for i, w in enumerate(all_wavs):
            combined_parts.append(w)
            if i < len(all_wavs) - 1:
                combined_parts.append(silence_gap)

        combined = np.concatenate(combined_parts)

        total_time = time.time() - t0
        print(f"[pipeline] Complete. STT: {t_stt:.2f}s, TTS: {total_tts:.2f}s, Total: {total_time:.2f}s")

        # Output
        if do_save and self.output_dir:
            basename = os.path.splitext(os.path.basename(input_file))[0]
            out_path = os.path.join(self.output_dir, f"{basename}_cloned.wav")
            self.audio_out.save(combined, sr, out_path)

        if do_play:
            print("[pipeline] Playing output...")
            self.audio_out.play_and_wait(combined, sr)
