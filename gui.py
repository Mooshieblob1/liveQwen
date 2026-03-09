#!/usr/bin/env python3
"""liveQwen GUI — Graphical interface for real-time voice cloning."""

import os
import sys
import glob
import time
import threading
import queue
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QSlider, QFileDialog,
    QTextEdit, QGroupBox, QSplitter, QFrame, QProgressBar, QLineEdit,
    QTabWidget, QSpacerItem, QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon, QTextCursor

# ── Ensure project root is importable ────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import (
    DEFAULT_STT_MODEL, DEFAULT_STT_DEVICE, DEFAULT_STT_COMPUTE,
    DEFAULT_TTS_MODEL, DEFAULT_TTS_MODEL_SMALL, DEFAULT_OUTPUT_DIR,
    SILENCE_THRESHOLD_S,
)


# ── Log redirect ─────────────────────────────────────────────
class LogSignal(QObject):
    """Thread-safe signal for redirecting stdout/stderr to GUI."""
    message = pyqtSignal(str)


class LogRedirector:
    """Redirects writes to a Qt signal."""

    def __init__(self, signal, original):
        self.signal = signal
        self.original = original

    def write(self, text):
        if text.strip():
            self.signal.emit(text.rstrip("\n"))
        self.original.write(text)

    def flush(self):
        self.original.flush()


# ── Worker thread for model loading / pipeline ───────────────
class PipelineWorker(QThread):
    """Runs the voice clone pipeline in a background thread."""
    log = pyqtSignal(str)
    status = pyqtSignal(str)
    models_loaded = pyqtSignal()
    utterance_detected = pyqtSignal(float)  # duration in seconds
    transcription_done = pyqtSignal(str, str, float)  # text, language, stt_time
    tts_done = pyqtSignal(float, float)  # tts_time, total_time
    pipeline_stopped = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._running = False
        self._pipeline = None
        self._stt = None
        self._tts = None
        self._audio_out = None
        self._virtual_mic = None

    def run(self):
        try:
            self._load_models()
            if not self._running:
                return
            self._run_pipeline()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self._cleanup()
            self.pipeline_stopped.emit()

    def _load_models(self):
        self._running = True
        cfg = self.config

        # STT
        self.status.emit("Loading STT model...")
        from stt import SpeechToText
        self._stt = SpeechToText(
            model_size=cfg["stt_model"],
            device=cfg["stt_device"],
            compute_type=cfg["stt_compute"],
        )

        # Auto-transcribe reference audio
        ref_text = cfg.get("ref_text") or None
        if ref_text is None:
            self.status.emit("Auto-transcribing reference audio...")
            segments = self._stt.transcribe_file(cfg["ref_audio"])
            ref_text = " ".join(seg["text"] for seg in segments)
            if ref_text.strip():
                self.log.emit(f'Reference transcript: "{ref_text}"')
            else:
                ref_text = ""
                self.log.emit("WARNING: Could not transcribe reference audio.")
        cfg["_ref_text"] = ref_text

        # TTS
        self.status.emit("Loading TTS model...")
        from tts import VoiceCloner
        self._tts = VoiceCloner(
            model_name=cfg["tts_model"],
            use_flash_attn=cfg["flash_attn"],
        )
        self._tts.setup_voice(ref_audio=cfg["ref_audio"], ref_text=ref_text)

        # Virtual mic
        if cfg["virtual_mic"]:
            self.status.emit("Setting up virtual microphone...")
            from audio_io import VirtualMic
            self._virtual_mic = VirtualMic()
            vmic_index = self._virtual_mic.setup()
            if vmic_index is not None:
                cfg["_mirror_device"] = str(vmic_index)
                cfg["_mirror_sink_name"] = "LiveQwenVoiceClone"
            else:
                self.log.emit("WARNING: Virtual mic setup failed.")
                cfg["_mirror_device"] = None
                cfg["_mirror_sink_name"] = None
        else:
            cfg["_mirror_device"] = cfg.get("mirror_device")
            cfg["_mirror_sink_name"] = None

        self.models_loaded.emit()
        self.status.emit("Models loaded — ready")

    def _run_pipeline(self):
        cfg = self.config
        from audio_io import AudioOutput
        from pipeline import LivePipeline

        self._audio_out = AudioOutput(
            output_device=cfg.get("output_device"),
            mirror_device=cfg.get("_mirror_device"),
            mirror_sink_name=cfg.get("_mirror_sink_name"),
        )

        self._pipeline = LivePipeline(
            stt=self._stt,
            tts=self._tts,
            audio_out=self._audio_out,
            input_device=cfg.get("input_device"),
            output_dir=cfg["output_dir"],
            do_play=cfg["play"],
            do_save=cfg["save"],
            silence_duration=cfg["silence"],
            push_to_talk=cfg["push_to_talk"],
        )

        self.status.emit("Pipeline running — listening...")
        self._pipeline.start()  # blocks until stopped

    def stop(self):
        self._running = False
        if self._pipeline:
            self._pipeline.stop()

    def trigger_ptt(self):
        """Trigger push-to-talk from the GUI."""
        if self._pipeline and hasattr(self._pipeline, '_detector'):
            detector = self._pipeline._detector
            if hasattr(detector, '_triggered'):
                detector._triggered.set()

    def _cleanup(self):
        if self._virtual_mic:
            self._virtual_mic.teardown()
            self._virtual_mic = None


# ── Main Window ──────────────────────────────────────────────
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("liveQwen — Voice Clone")
        self.setMinimumSize(900, 700)
        self._worker = None
        self._is_running = False

        # Log redirect
        self._log_signal = LogSignal()
        self._log_signal.message.connect(self._append_log)

        self._build_ui()
        self._populate_devices()
        self._populate_ref_voices()

    # ── UI Construction ──────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8)

        # Title
        title = QLabel("liveQwen — Real-Time Voice Clone")
        title.setFont(QFont("", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Tabs
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Tab 1: Settings
        settings_widget = QWidget()
        tabs.addTab(settings_widget, "Settings")
        self._build_settings_tab(settings_widget)

        # Tab 2: Live Session
        session_widget = QWidget()
        tabs.addTab(session_widget, "Live Session")
        self._build_session_tab(session_widget)

        self._tabs = tabs

    def _build_settings_tab(self, parent):
        layout = QVBoxLayout(parent)

        # ── Reference Voice ──────────────────────────────────
        ref_group = QGroupBox("Reference Voice")
        ref_layout = QVBoxLayout(ref_group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Reference Audio:"))
        self._ref_audio_combo = QComboBox()
        self._ref_audio_combo.setMinimumWidth(300)
        row.addWidget(self._ref_audio_combo, 1)
        self._ref_browse_btn = QPushButton("Browse...")
        self._ref_browse_btn.clicked.connect(self._browse_ref_audio)
        row.addWidget(self._ref_browse_btn)
        ref_layout.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Reference Text:"))
        self._ref_text_input = QLineEdit()
        self._ref_text_input.setPlaceholderText("(leave blank for auto-transcription)")
        row2.addWidget(self._ref_text_input, 1)
        ref_layout.addLayout(row2)

        layout.addWidget(ref_group)

        # ── Audio Devices ────────────────────────────────────
        dev_group = QGroupBox("Audio Devices")
        dev_layout = QVBoxLayout(dev_group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Input (mic):"))
        self._input_device_combo = QComboBox()
        row.addWidget(self._input_device_combo, 1)
        dev_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Output:"))
        self._output_device_combo = QComboBox()
        row.addWidget(self._output_device_combo, 1)
        dev_layout.addLayout(row)

        layout.addWidget(dev_group)

        # ── Models ───────────────────────────────────────────
        model_group = QGroupBox("Models")
        model_layout = QVBoxLayout(model_group)

        row = QHBoxLayout()
        row.addWidget(QLabel("STT Model:"))
        self._stt_model_combo = QComboBox()
        self._stt_model_combo.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self._stt_model_combo.setCurrentText(DEFAULT_STT_MODEL)
        row.addWidget(self._stt_model_combo)

        row.addWidget(QLabel("  Device:"))
        self._stt_device_combo = QComboBox()
        self._stt_device_combo.addItems(["cpu", "cuda"])
        self._stt_device_combo.setCurrentText(DEFAULT_STT_DEVICE)
        row.addWidget(self._stt_device_combo)

        row.addWidget(QLabel("  Compute:"))
        self._stt_compute_combo = QComboBox()
        self._stt_compute_combo.addItems(["int8", "int8_float16", "float16", "float32"])
        self._stt_compute_combo.setCurrentText(DEFAULT_STT_COMPUTE)
        row.addWidget(self._stt_compute_combo)

        row.addStretch()
        model_layout.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("TTS Model:"))
        self._tts_model_combo = QComboBox()
        self._tts_model_combo.addItems([DEFAULT_TTS_MODEL, DEFAULT_TTS_MODEL_SMALL])
        self._tts_model_combo.setCurrentText(DEFAULT_TTS_MODEL)
        row2.addWidget(self._tts_model_combo)

        self._flash_attn_check = QCheckBox("Flash Attention")
        self._flash_attn_check.setChecked(True)
        row2.addWidget(self._flash_attn_check)

        row2.addStretch()
        model_layout.addLayout(row2)

        layout.addWidget(model_group)

        # ── Pipeline Options ─────────────────────────────────
        opts_group = QGroupBox("Pipeline Options")
        opts_layout = QVBoxLayout(opts_group)

        row = QHBoxLayout()
        self._push_to_talk_check = QCheckBox("Push-to-Talk")
        self._push_to_talk_check.setChecked(True)
        row.addWidget(self._push_to_talk_check)

        self._virtual_mic_check = QCheckBox("Virtual Mic (Discord/OBS)")
        self._virtual_mic_check.setChecked(True)
        row.addWidget(self._virtual_mic_check)

        self._play_check = QCheckBox("Play through speakers")
        self._play_check.setChecked(True)
        row.addWidget(self._play_check)

        self._save_check = QCheckBox("Save to files")
        self._save_check.setChecked(True)
        row.addWidget(self._save_check)

        row.addStretch()
        opts_layout.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Silence threshold (seconds):"))
        self._silence_slider = QSlider(Qt.Orientation.Horizontal)
        self._silence_slider.setRange(5, 50)  # 0.5 to 5.0
        self._silence_slider.setValue(int(SILENCE_THRESHOLD_S * 10))
        self._silence_slider.setTickInterval(5)
        self._silence_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._silence_slider.valueChanged.connect(self._update_silence_label)
        row2.addWidget(self._silence_slider)
        self._silence_label = QLabel(f"{SILENCE_THRESHOLD_S:.1f}s")
        self._silence_label.setMinimumWidth(35)
        row2.addWidget(self._silence_label)
        opts_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Output directory:"))
        self._output_dir_input = QLineEdit(DEFAULT_OUTPUT_DIR)
        row3.addWidget(self._output_dir_input, 1)
        browse_dir_btn = QPushButton("Browse...")
        browse_dir_btn.clicked.connect(self._browse_output_dir)
        row3.addWidget(browse_dir_btn)
        opts_layout.addLayout(row3)

        layout.addWidget(opts_group)
        layout.addStretch()

    def _build_session_tab(self, parent):
        layout = QVBoxLayout(parent)

        # Status bar
        status_frame = QFrame()
        status_frame.setFrameShape(QFrame.Shape.StyledPanel)
        status_layout = QHBoxLayout(status_frame)
        self._status_indicator = QLabel("●")
        self._status_indicator.setFont(QFont("", 16))
        self._status_indicator.setStyleSheet("color: gray;")
        status_layout.addWidget(self._status_indicator)
        self._status_label = QLabel("Stopped")
        self._status_label.setFont(QFont("", 12, QFont.Weight.Bold))
        status_layout.addWidget(self._status_label, 1)
        layout.addWidget(status_frame)

        # Controls
        ctrl_layout = QHBoxLayout()

        self._start_btn = QPushButton("▶  Start")
        self._start_btn.setFont(QFont("", 12, QFont.Weight.Bold))
        self._start_btn.setMinimumHeight(50)
        self._start_btn.setStyleSheet(
            "QPushButton { background-color: #2d8a4e; color: white; border-radius: 8px; padding: 8px 24px; }"
            "QPushButton:hover { background-color: #3ba55d; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        self._start_btn.clicked.connect(self._on_start)
        ctrl_layout.addWidget(self._start_btn)

        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setFont(QFont("", 12, QFont.Weight.Bold))
        self._stop_btn.setMinimumHeight(50)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(
            "QPushButton { background-color: #d73a49; color: white; border-radius: 8px; padding: 8px 24px; }"
            "QPushButton:hover { background-color: #e05561; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        self._stop_btn.clicked.connect(self._on_stop)
        ctrl_layout.addWidget(self._stop_btn)

        self._ptt_btn = QPushButton("🎤  Push to Talk")
        self._ptt_btn.setFont(QFont("", 12, QFont.Weight.Bold))
        self._ptt_btn.setMinimumHeight(50)
        self._ptt_btn.setEnabled(False)
        self._ptt_btn.setStyleSheet(
            "QPushButton { background-color: #0969da; color: white; border-radius: 8px; padding: 8px 24px; }"
            "QPushButton:hover { background-color: #218bff; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        self._ptt_btn.clicked.connect(self._on_ptt)
        ctrl_layout.addWidget(self._ptt_btn)

        layout.addLayout(ctrl_layout)

        # Transcription log
        log_group = QGroupBox("Session Log")
        log_layout = QVBoxLayout(log_group)
        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFont(QFont("Monospace", 10))
        self._log_view.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; border: 1px solid #444;")
        log_layout.addWidget(self._log_view)

        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self._log_view.clear)
        log_layout.addWidget(clear_btn, alignment=Qt.AlignmentFlag.AlignRight)

        layout.addWidget(log_group, 1)

    # ── Populate UI ──────────────────────────────────────────
    def _populate_devices(self):
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            self._input_device_combo.addItem("(System Default)", None)
            self._output_device_combo.addItem("(System Default)", None)
            for i, d in enumerate(devices):
                label = f"[{i}] {d['name']}"
                if d["max_input_channels"] > 0:
                    self._input_device_combo.addItem(label, i)
                if d["max_output_channels"] > 0:
                    self._output_device_combo.addItem(label, i)
        except Exception:
            self._input_device_combo.addItem("(Default)", None)
            self._output_device_combo.addItem("(Default)", None)

    def _populate_ref_voices(self):
        voice_dir = os.path.join(SCRIPT_DIR, "voice_samples")
        wavs = sorted(glob.glob(os.path.join(voice_dir, "*.wav")))
        for w in wavs:
            self._ref_audio_combo.addItem(os.path.basename(w), w)
        if not wavs:
            self._ref_audio_combo.addItem("(no WAV files in voice_samples/)", "")

    # ── Slots ────────────────────────────────────────────────
    def _update_silence_label(self, value):
        self._silence_label.setText(f"{value / 10:.1f}s")

    def _browse_ref_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Audio", "", "WAV Files (*.wav);;All Files (*)"
        )
        if path:
            idx = self._ref_audio_combo.findData(path)
            if idx < 0:
                self._ref_audio_combo.addItem(os.path.basename(path), path)
                idx = self._ref_audio_combo.count() - 1
            self._ref_audio_combo.setCurrentIndex(idx)

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self._output_dir_input.setText(path)

    def _on_start(self):
        ref_audio = self._ref_audio_combo.currentData()
        if not ref_audio or not os.path.isfile(ref_audio):
            self._append_log("ERROR: Please select a valid reference audio file.")
            return

        config = {
            "ref_audio": ref_audio,
            "ref_text": self._ref_text_input.text().strip() or None,
            "stt_model": self._stt_model_combo.currentText(),
            "stt_device": self._stt_device_combo.currentText(),
            "stt_compute": self._stt_compute_combo.currentText(),
            "tts_model": self._tts_model_combo.currentText(),
            "flash_attn": self._flash_attn_check.isChecked(),
            "push_to_talk": self._push_to_talk_check.isChecked(),
            "virtual_mic": self._virtual_mic_check.isChecked(),
            "play": self._play_check.isChecked(),
            "save": self._save_check.isChecked(),
            "silence": self._silence_slider.value() / 10.0,
            "output_dir": self._output_dir_input.text(),
            "input_device": self._input_device_combo.currentData(),
            "output_device": self._output_device_combo.currentData(),
            "mirror_device": None,
        }

        # Redirect stdout/stderr
        sys.stdout = LogRedirector(self._log_signal.message, sys.__stdout__)
        sys.stderr = LogRedirector(self._log_signal.message, sys.__stderr__)

        self._set_running(True)
        self._tabs.setCurrentIndex(1)

        self._worker = PipelineWorker(config)
        self._worker.log.connect(self._append_log)
        self._worker.status.connect(self._update_status)
        self._worker.models_loaded.connect(self._on_models_loaded)
        self._worker.pipeline_stopped.connect(self._on_pipeline_stopped)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_stop(self):
        if self._worker:
            self._update_status("Stopping...")
            self._worker.stop()

    def _on_ptt(self):
        if self._worker:
            self._worker.trigger_ptt()
            self._append_log("🎤 Push-to-talk triggered")

    def _on_models_loaded(self):
        if self._push_to_talk_check.isChecked():
            self._ptt_btn.setEnabled(True)

    def _on_pipeline_stopped(self):
        self._set_running(False)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def _on_error(self, msg):
        self._append_log(f"ERROR: {msg}")
        self._set_running(False)

    # ── Helpers ──────────────────────────────────────────────
    def _set_running(self, running: bool):
        self._is_running = running
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        if not running:
            self._ptt_btn.setEnabled(False)
            self._status_indicator.setStyleSheet("color: gray;")
            self._status_label.setText("Stopped")
        else:
            self._status_indicator.setStyleSheet("color: #3ba55d;")

    def _update_status(self, text: str):
        self._status_label.setText(text)

    def _append_log(self, text: str):
        self._log_view.append(text)
        # Auto-scroll to bottom
        cursor = self._log_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._log_view.setTextCursor(cursor)

    def closeEvent(self, event):
        if self._worker and self._is_running:
            self._worker.stop()
            self._worker.wait(5000)
        event.accept()


# ── Entry point ──────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)

    # Dark theme
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(212, 212, 212))
    palette.setColor(QPalette.ColorRole.Base, QColor(37, 37, 37))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(50, 50, 50))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(212, 212, 212))
    palette.setColor(QPalette.ColorRole.Text, QColor(212, 212, 212))
    palette.setColor(QPalette.ColorRole.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(212, 212, 212))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(128, 128, 128))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(128, 128, 128))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
