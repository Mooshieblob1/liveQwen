import os
import queue
import subprocess
import sys
import termios
import threading
import tty
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

from config import SAMPLE_RATE, CHANNELS, CHUNK_SIZE

VIRTUAL_SINK_NAME = "LiveQwenVoiceClone"


def list_devices():
    """Print all available audio devices."""
    print(sd.query_devices())


def resolve_device(device_arg):
    """Resolve a device argument (name substring or integer index) to an index."""
    if device_arg is None:
        return None
    try:
        return int(device_arg)
    except ValueError:
        pass
    # Search by name substring
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if device_arg.lower() in d["name"].lower():
            return i
    raise ValueError(f"Audio device not found: {device_arg}")


class MicrophoneStream:
    """Captures audio from a microphone in a background thread via sounddevice.

    Audio chunks are placed into a queue for consumption by the pipeline.
    Utterance segmentation (VAD) is handled downstream.
    """

    def __init__(self, device=None, sample_rate=SAMPLE_RATE, channels=CHANNELS,
                 chunk_size=CHUNK_SIZE):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device = resolve_device(device)
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[mic] {status}")
        # indata is (frames, channels) float32 — copy to decouple from buffer
        self._queue.put(indata[:, 0].copy())

    def start(self):
        self._stream = sd.InputStream(
            device=self.device,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=self._callback,
        )
        self._stream.start()

    def read(self, timeout=None) -> np.ndarray | None:
        """Return the next audio chunk, or None if the stream has been stopped."""
        return self._queue.get(timeout=timeout)

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        # Signal consumers
        self._queue.put(None)


class UtteranceDetector:
    """Accumulates audio chunks and yields complete utterances using Silero VAD.

    Uses the Silero VAD neural network for robust speech detection — much more
    accurate than energy-based thresholds at handling quiet consonants, breathing,
    and natural pauses between words.
    """

    # Silero VAD requires 512 samples at 16kHz (32ms chunks)
    SILERO_CHUNK = 512

    def __init__(self, sample_rate=SAMPLE_RATE, silence_duration=2.0,
                 speech_threshold=0.3, min_utterance_duration=0.3,
                 max_utterance_duration=30.0):
        self.sample_rate = sample_rate
        self.silence_duration = silence_duration
        self.speech_threshold = speech_threshold
        self.min_utterance_samples = int(min_utterance_duration * sample_rate)
        self.max_utterance_samples = int(max_utterance_duration * sample_rate)

        # Load Silero VAD
        self._vad_model, _utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True,
        )

        self._buffer: list[np.ndarray] = []
        self._silence_samples = 0
        self._speech_detected = False
        self._total_samples = 0
        self._pending: list[np.ndarray] = []  # accumulates samples until we have a full SILERO_CHUNK

    def _is_speech(self, chunk_512: np.ndarray) -> bool:
        """Run Silero VAD on exactly 512 samples, return True if speech detected."""
        tensor = torch.from_numpy(chunk_512).float()
        prob = self._vad_model(tensor, self.sample_rate).item()
        return prob > self.speech_threshold

    def feed(self, chunk: np.ndarray) -> np.ndarray | None:
        """Feed an audio chunk. Returns a complete utterance array when detected, else None."""
        # Accumulate into pending buffer, process in 512-sample slices
        self._pending.append(chunk)
        pending_audio = np.concatenate(self._pending)

        result = None
        offset = 0
        while offset + self.SILERO_CHUNK <= len(pending_audio):
            sl = pending_audio[offset:offset + self.SILERO_CHUNK]
            offset += self.SILERO_CHUNK
            r = self._process_slice(sl)
            if r is not None:
                result = r

        # Keep leftover samples for next call
        if offset < len(pending_audio):
            self._pending = [pending_audio[offset:]]
        else:
            self._pending = []

        return result

    def _process_slice(self, chunk_512: np.ndarray) -> np.ndarray | None:
        """Process a single 512-sample slice through VAD logic."""
        is_speech = self._is_speech(chunk_512)

        if is_speech:
            self._speech_detected = True
            self._silence_samples = 0
            self._buffer.append(chunk_512)
            self._total_samples += len(chunk_512)
        elif self._speech_detected:
            self._buffer.append(chunk_512)
            self._total_samples += len(chunk_512)
            self._silence_samples += len(chunk_512)

            if self._silence_samples >= int(self.silence_duration * self.sample_rate):
                return self._emit()
        # else: silence before any speech — discard

        # Force-emit if utterance exceeds max duration
        if self._speech_detected and self._total_samples >= self.max_utterance_samples:
            return self._emit()

        return None

    def _emit(self) -> np.ndarray | None:
        """Emit the buffered utterance and reset state."""
        utterance = np.concatenate(self._buffer)
        self._buffer.clear()
        self._silence_samples = 0
        self._speech_detected = False
        self._total_samples = 0
        # Reset VAD state for next utterance
        self._vad_model.reset_states()

        if len(utterance) >= self.min_utterance_samples:
            return utterance
        return None

    def flush(self) -> np.ndarray | None:
        """Flush any remaining buffered audio as a final utterance."""
        if self._buffer:
            return self._emit()
        return None


class PushToTalkDetector:
    """Accumulates all audio until the user presses Enter to trigger processing.

    No VAD — just records everything between start and Enter press.
    """

    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._buffer: list[np.ndarray] = []
        self._triggered = threading.Event()
        self._listener_thread = None
        self._running = False

    def start(self):
        """Start listening for Enter key presses in a background thread."""
        self._running = True
        self._listener_thread = threading.Thread(target=self._listen, daemon=True)
        self._listener_thread.start()

    def _listen(self):
        """Block on stdin, set trigger on each Enter press.

        Reads raw bytes to handle terminals using the kitty keyboard
        protocol (which sends ESC sequences instead of plain '\n').
        """
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            buf = b""
            while self._running:
                ch = os.read(fd, 1)
                if not ch:
                    break
                # Plain Enter (\r or \n)
                if ch in (b"\r", b"\n"):
                    self._triggered.set()
                    buf = b""
                    continue
                # Ctrl+C
                if ch == b"\x03":
                    # Restore terminal first, then raise
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    os.kill(os.getpid(), 2)  # SIGINT
                    return
                # Accumulate ESC sequences (kitty protocol sends \x1b[13u for Enter)
                buf += ch
                if buf.endswith(b"u") and b"\x1b[13" in buf:
                    # Kitty-encoded Enter
                    self._triggered.set()
                    buf = b""
                elif len(buf) > 20:
                    buf = b""  # discard junk
        except (EOFError, OSError):
            pass
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except (termios.error, OSError):
                pass

    def stop(self):
        self._running = False

    def feed(self, chunk: np.ndarray) -> np.ndarray | None:
        """Buffer audio. Return utterance when Enter was pressed, else None."""
        self._buffer.append(chunk)

        if self._triggered.is_set():
            self._triggered.clear()
            if self._buffer:
                utterance = np.concatenate(self._buffer)
                self._buffer.clear()
                min_samples = int(0.3 * self.sample_rate)
                if len(utterance) >= min_samples:
                    return utterance
        return None

    def flush(self) -> np.ndarray | None:
        if self._buffer:
            utterance = np.concatenate(self._buffer)
            self._buffer.clear()
            if len(utterance) >= int(0.3 * self.sample_rate):
                return utterance
        return None


class VirtualMic:
    """Manages a PulseAudio virtual microphone sink.

    Creates a null-sink whose monitor source appears as a recording device
    (microphone) in apps like Discord and OBS.
    """

    def __init__(self, sink_name=VIRTUAL_SINK_NAME):
        self.sink_name = sink_name
        self._module_ids = []  # track all loaded PA modules for cleanup

    @staticmethod
    def _ensure_pactl() -> bool:
        """Make sure pactl is available, installing it if necessary.

        Returns True if pactl is usable, False otherwise.
        """
        # Fast path: already installed
        try:
            subprocess.run(["pactl", "--version"], capture_output=True, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        print("[virtual-mic] pactl not found — attempting to install PulseAudio utilities...")

        # Detect package manager and install
        installers = [
            # Debian / Ubuntu / Mint / Pop!_OS
            (["apt-get", "--version"], ["sudo", "apt-get", "install", "-y", "pulseaudio-utils"]),
            # Fedora / RHEL / CentOS
            (["dnf", "--version"],     ["sudo", "dnf", "install", "-y", "pulseaudio-utils"]),
            # Older RHEL / CentOS
            (["yum", "--version"],     ["sudo", "yum", "install", "-y", "pulseaudio-utils"]),
            # Arch / Manjaro
            (["pacman", "--version"],  ["sudo", "pacman", "-S", "--noconfirm", "libpulse"]),
            # openSUSE
            (["zypper", "--version"],  ["sudo", "zypper", "install", "-y", "pulseaudio-utils"]),
        ]

        for check_cmd, install_cmd in installers:
            try:
                subprocess.run(check_cmd, capture_output=True, check=True)
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue

            # Found a package manager — run the install
            print(f"[virtual-mic] Running: {' '.join(install_cmd)}")
            try:
                subprocess.run(install_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[virtual-mic] Install failed (exit {e.returncode}). "
                      "Try installing PulseAudio utilities manually.")
                return False

            # Verify it worked
            try:
                subprocess.run(["pactl", "--version"], capture_output=True, check=True)
                print("[virtual-mic] PulseAudio utilities installed successfully.")
                return True
            except (FileNotFoundError, subprocess.CalledProcessError):
                print("[virtual-mic] pactl still not found after install.")
                return False

        print("[virtual-mic] Could not detect a supported package manager "
              "(apt, dnf, yum, pacman, zypper). Install PulseAudio utilities manually.")
        return False

    def setup(self) -> int | None:
        """Create the virtual sink. Returns the device index or None on failure."""
        if not self._ensure_pactl():
            return None

        # Remove any existing sink with the same name
        self.teardown()

        # Create null sink (48 kHz mono — widely compatible with PortAudio)
        try:
            result = subprocess.run(
                ["pactl", "load-module", "module-null-sink",
                 f"sink_name={self.sink_name}",
                 "rate=48000",
                 "channels=1",
                 f'sink_properties=device.description="liveQwen Voice Clone"'],
                capture_output=True, text=True, check=True,
            )
            self._module_ids.append(result.stdout.strip())
            print(f"[virtual-mic] Created virtual sink '{self.sink_name}'")
        except subprocess.CalledProcessError as e:
            print(f"[virtual-mic] Failed to create sink: {e.stderr}")
            return None

        # Create a remapped source from the monitor so it appears as a real
        # microphone in apps like Discord (which hide monitor sources)
        remap_source_name = f"{self.sink_name}_Mic"
        try:
            result = subprocess.run(
                ["pactl", "load-module", "module-remap-source",
                 f"source_name={remap_source_name}",
                 f"master={self.sink_name}.monitor",
                 f'source_properties=device.description="liveQwen Microphone"'],
                capture_output=True, text=True, check=True,
            )
            self._module_ids.append(result.stdout.strip())
            print(f"[virtual-mic] Created virtual microphone '{remap_source_name}'")
            print(f"[virtual-mic] Select 'liveQwen Microphone' as your input device in Discord/OBS.")
        except subprocess.CalledProcessError as e:
            # Not fatal — the monitor source still works in some apps
            print(f"[virtual-mic] Note: Could not create remapped source: {e.stderr}")
            print(f"[virtual-mic] Use 'Monitor of liveQwen Voice Clone' as mic input instead.")

        # Find the device index in sounddevice
        return self._find_device_index()

    def _find_device_index(self) -> int | None:
        """Find the sounddevice index for our virtual sink."""
        import time

        # Force sounddevice to re-query the system device list
        sd._terminate()
        sd._initialize()

        # Retry a few times — PA may take a moment to register
        for attempt in range(5):
            time.sleep(0.5)
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                name_lower = d["name"].lower()
                # Match by sink name or the human-readable description
                if (self.sink_name.lower() in name_lower
                        or "liveqwen" in name_lower) \
                        and d["max_output_channels"] > 0:
                    print(f"[virtual-mic] Found device index {i}: {d['name']}")
                    return i
            # Re-initialize on each retry to pick up newly registered devices
            if attempt < 4:
                sd._terminate()
                sd._initialize()

        # Debug: show what devices we can see
        print(f"[virtual-mic] Warning: Could not find device index for '{self.sink_name}'")
        print("[virtual-mic] Available output devices:")
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d["max_output_channels"] > 0:
                print(f"  [{i}] {d['name']}")
        return None

    def teardown(self):
        """Remove all virtual audio modules."""
        # Unload in reverse order (remap source first, then sink)
        for mid in reversed(self._module_ids):
            try:
                subprocess.run(
                    ["pactl", "unload-module", mid],
                    capture_output=True, check=True,
                )
            except subprocess.CalledProcessError:
                pass
        if self._module_ids:
            print(f"[virtual-mic] Removed virtual audio devices ({len(self._module_ids)} modules)")
            self._module_ids.clear()
        else:
            # Fallback: clean up by sink name
            try:
                result = subprocess.run(
                    ["pactl", "list", "short", "modules"],
                    capture_output=True, text=True, check=True,
                )
                for line in result.stdout.splitlines():
                    if self.sink_name in line:
                        module_id = line.split()[0]
                        subprocess.run(
                            ["pactl", "unload-module", module_id],
                            capture_output=True, check=True,
                        )
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass


class AudioOutput:
    """Handles audio output to speakers, files, and virtual devices."""

    def __init__(self, output_device=None, mirror_device=None, mirror_sink_name=None):
        self.output_device = resolve_device(output_device)
        self.mirror_device = resolve_device(mirror_device) if mirror_device is not None else None
        self._mirror_sink_name = mirror_sink_name  # PulseAudio sink name for paplay
        # Pre-generate a short ready beep (880Hz, 0.15s, very quiet)
        t = np.linspace(0, 0.15, int(0.15 * 16000), endpoint=False, dtype=np.float32)
        self._beep = 0.05 * np.sin(2 * np.pi * 880 * t)
        self._beep_sr = 16000

    @staticmethod
    def _play_stream(audio: np.ndarray, sample_rate: int, device) -> None:
        """Play audio through an explicit OutputStream (thread-safe)."""
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            device=device,
            dtype='float32',
        )
        stream.start()
        stream.write(audio.astype(np.float32))
        stream.stop()
        stream.close()

    def play_ready_beep(self):
        """Play a short beep to indicate readiness (only on primary device)."""
        self._play_stream(self._beep, self._beep_sr, self.output_device)

    def play(self, audio: np.ndarray, sample_rate: int, blocking=False):
        """Play audio through the primary output device."""
        if blocking:
            self._play_stream(audio, sample_rate, self.output_device)
        else:
            t = threading.Thread(
                target=self._play_stream, args=(audio, sample_rate, self.output_device),
                daemon=True,
            )
            t.start()

    @staticmethod
    def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """Simple linear-interpolation resample (no extra dependencies)."""
        if src_rate == dst_rate:
            return audio
        duration = len(audio) / src_rate
        dst_len = int(duration * dst_rate)
        indices = np.linspace(0, len(audio) - 1, dst_len, dtype=np.float32)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def play_mirror(self, audio: np.ndarray, sample_rate: int):
        """Play audio to the mirror device via paplay subprocess (avoids PortAudio conflicts)."""
        if self.mirror_device is None:
            return

        sink = self._mirror_sink_name
        if sink is None:
            # Fall back to device name from sounddevice
            try:
                dev_info = sd.query_devices(self.mirror_device)
                sink = dev_info["name"]
            except Exception as e:
                print(f"[output] Mirror device error: {e}")
                return

        try:
            # Convert to 16-bit PCM
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

            proc = subprocess.Popen(
                ["paplay",
                 "--device", sink,
                 "--format=s16le",
                 "--channels=1",
                 f"--rate={sample_rate}",
                 "--raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            proc.stdin.write(pcm)
            proc.stdin.close()
            proc.wait(timeout=30)
            if proc.returncode != 0:
                err = proc.stderr.read().decode(errors="replace").strip()
                print(f"[output] Mirror paplay error: {err}")
        except FileNotFoundError:
            print("[output] paplay not found — mirror output disabled.")
            self.mirror_device = None
        except Exception as e:
            print(f"[output] Mirror device error: {e}")

    def save(self, audio: np.ndarray, sample_rate: int, path: str):
        """Save audio to a WAV file."""
        sf.write(path, audio, sample_rate)
        print(f"[output] Saved: {path}")

    def play_and_wait(self, audio: np.ndarray, sample_rate: int):
        """Play audio on primary device (blocking) and mirror device simultaneously."""
        # Start mirror playback in background thread if configured
        mirror_thread = None
        if self.mirror_device is not None:
            mirror_thread = threading.Thread(
                target=self.play_mirror, args=(audio, sample_rate), daemon=True
            )
            mirror_thread.start()

        # Primary playback (blocking) — uses its own OutputStream, no conflict
        self._play_stream(audio, sample_rate, self.output_device)

        # Wait for mirror to finish
        if mirror_thread is not None:
            mirror_thread.join(timeout=30)
