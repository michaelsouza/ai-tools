#!/usr/bin/env python3
"""
GUI for Real-time Audio Transcription using Faster-Whisper

A graphical interface for real-time speech-to-text transcription using the
Faster-Whisper model running locally. Features include model selection,
start/stop recording, copy to clipboard, and clear text.

Usage:
    python audio_transcribe_gui.py

Requirements:
    - customtkinter: Modern GUI framework
    - faster-whisper: Local transcription engine
    - sounddevice: Microphone audio capture
    - numpy: Audio buffer handling
"""

import threading
import queue
from typing import Optional, Tuple

# Check for required dependencies
try:
    import customtkinter as ctk
except ImportError:
    print("Error: customtkinter not installed.")
    print("Install with: pip install customtkinter")
    exit(2)

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Error: faster-whisper not installed.")
    print("Install with: pip install faster-whisper")
    exit(2)

try:
    import sounddevice as sd
except ImportError:
    print("Error: sounddevice not installed.")
    print("Install with: pip install sounddevice")
    exit(2)

try:
    import numpy as np
except ImportError:
    print("Error: numpy not installed.")
    print("Install with: pip install numpy")
    exit(2)

# Constants
MODEL_SIZES = ['tiny', 'base', 'small', 'medium', 'large-v3']
LANGUAGES = {'English': 'en', 'Português': 'pt'}
DEFAULT_SAMPLE_RATE = 16000

# VAD parameters
AUDIO_BLOCK_DURATION = 0.1    # Small blocks for granular silence detection (100ms)
MIN_CHUNK_DURATION = 1.0      # Minimum audio before sending (seconds)
MAX_CHUNK_DURATION = 5.0      # Force send even without silence (seconds)
SILENCE_THRESHOLD = 0.01      # RMS energy threshold for silence detection
SILENCE_DURATION = 0.3        # Required silence duration to trigger send (seconds)

# Colors
RECORDING_COLOR = "#dc3545"
READY_COLOR = "#28a745"
LOADING_COLOR = "#ffc107"


def get_audio_device_info() -> Optional[Tuple[int, int]]:
    """
    Detect default microphone and return (device_id, sample_rate).
    Returns None if no input device available.
    """
    try:
        default_input = sd.query_devices(kind='input')
        if default_input is None or default_input.get('max_input_channels', 0) == 0:
            return None
        device_id = default_input.get('index', 0)
        return device_id, DEFAULT_SAMPLE_RATE
    except Exception:
        return None


def detect_device_and_compute_type() -> Tuple[str, str]:
    """
    Auto-detect device and compute type.
    Returns (device, compute_type) tuple.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda', 'float16'
    except ImportError:
        pass
    return 'cpu', 'int8'


class TranscriptionApp(ctk.CTk):
    """Main application window for audio transcription."""

    def __init__(self):
        super().__init__()

        # Configure window
        self.title("Audio Transcription")
        self.geometry("700x500")
        self.minsize(500, 400)

        # Set appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # State variables
        self.model: Optional[WhisperModel] = None
        self.is_recording = False
        self.audio_queue: queue.Queue = queue.Queue()
        self.text_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.audio_thread: Optional[threading.Thread] = None
        self.transcription_thread: Optional[threading.Thread] = None

        # Detect audio device
        self.audio_info = get_audio_device_info()

        # Build UI
        self._create_widgets()

        # Start text update loop
        self._update_text_from_queue()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _create_widgets(self):
        """Create all GUI widgets."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Top frame (model selection + status)
        top_frame = ctk.CTkFrame(self)
        top_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        top_frame.grid_columnconfigure(3, weight=1)

        # Model selection
        model_label = ctk.CTkLabel(top_frame, text="Model:")
        model_label.grid(row=0, column=0, padx=(10, 5), pady=10)

        self.model_var = ctk.StringVar(value="base")
        self.model_dropdown = ctk.CTkOptionMenu(
            top_frame,
            values=MODEL_SIZES,
            variable=self.model_var,
            width=120
        )
        self.model_dropdown.grid(row=0, column=1, padx=5, pady=10, sticky="w")

        # Language selection
        lang_label = ctk.CTkLabel(top_frame, text="Idioma:")
        lang_label.grid(row=0, column=2, padx=(20, 5), pady=10)

        self.lang_var = ctk.StringVar(value="Português")
        self.lang_dropdown = ctk.CTkOptionMenu(
            top_frame,
            values=list(LANGUAGES.keys()),
            variable=self.lang_var,
            width=120
        )
        self.lang_dropdown.grid(row=0, column=3, padx=5, pady=10, sticky="w")

        # Status indicator
        status_frame = ctk.CTkFrame(top_frame, fg_color="transparent")
        status_frame.grid(row=0, column=4, padx=10, pady=10, sticky="e")

        self.status_dot = ctk.CTkLabel(
            status_frame,
            text="●",
            font=ctk.CTkFont(size=16),
            text_color=READY_COLOR if self.audio_info else RECORDING_COLOR
        )
        self.status_dot.grid(row=0, column=0, padx=(0, 5))

        status_text = "Ready" if self.audio_info else "No Microphone"
        self.status_label = ctk.CTkLabel(status_frame, text=status_text)
        self.status_label.grid(row=0, column=1)

        # Text area
        self.text_box = ctk.CTkTextbox(self, wrap="word", font=ctk.CTkFont(size=14))
        self.text_box.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        # Bottom frame (buttons)
        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.grid(row=2, column=0, padx=10, pady=(5, 10), sticky="ew")
        bottom_frame.grid_columnconfigure((0, 1, 2), weight=1)

        # Start/Stop button
        self.start_button = ctk.CTkButton(
            bottom_frame,
            text="● Iniciar",
            command=self._toggle_recording,
            width=140,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.start_button.grid(row=0, column=0, padx=10, pady=10)

        if not self.audio_info:
            self.start_button.configure(state="disabled")

        # Copy button
        self.copy_button = ctk.CTkButton(
            bottom_frame,
            text="Copiar",
            command=self._copy_to_clipboard,
            width=140,
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.copy_button.grid(row=0, column=1, padx=10, pady=10)

        # Clear button
        self.clear_button = ctk.CTkButton(
            bottom_frame,
            text="Limpar",
            command=self._clear_text,
            width=140,
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.clear_button.grid(row=0, column=2, padx=10, pady=10)

    def _toggle_recording(self):
        """Start or stop recording based on current state."""
        if self.is_recording:
            self._stop_transcription()
        else:
            self._start_transcription()

    def _start_transcription(self):
        """Start audio capture and transcription."""
        if self.audio_info is None:
            self._set_status("No Microphone", RECORDING_COLOR)
            return

        # Disable model/language selection during recording
        self.model_dropdown.configure(state="disabled")
        self.lang_dropdown.configure(state="disabled")
        self.start_button.configure(state="disabled")

        # Load model if needed
        model_size = self.model_var.get()
        if self.model is None or not hasattr(self, '_loaded_model_size') or self._loaded_model_size != model_size:
            self._set_status("Loading...", LOADING_COLOR)
            self.update()

            # Load model in background
            load_thread = threading.Thread(target=self._load_model, args=(model_size,))
            load_thread.start()
        else:
            self._begin_recording()

    def _load_model(self, model_size: str):
        """Load the transcription model in background thread."""
        try:
            device, compute_type = detect_device_and_compute_type()
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            self._loaded_model_size = model_size

            # Start recording after model loads
            self.after(0, self._begin_recording)
        except Exception as e:
            self.after(0, lambda: self._handle_model_error(str(e)))

    def _handle_model_error(self, error_msg: str):
        """Handle model loading error."""
        self._set_status("Error", RECORDING_COLOR)
        self.model_dropdown.configure(state="normal")
        self.lang_dropdown.configure(state="normal")
        self.start_button.configure(state="normal")

        # Show error in text box
        self.text_box.insert("end", f"[Error loading model: {error_msg}]\n")

    def _begin_recording(self):
        """Begin the actual recording after model is loaded."""
        self.is_recording = True
        self.stop_event.clear()

        # Clear queues
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait()
            except queue.Empty:
                break

        # Update UI
        self._set_status("Recording", RECORDING_COLOR)
        self.start_button.configure(text="■ Parar", state="normal")

        # Get selected language code
        lang_code = LANGUAGES.get(self.lang_var.get(), 'pt')

        # Start threads
        device_id, sample_rate = self.audio_info
        self.audio_thread = threading.Thread(
            target=self._audio_capture_thread,
            args=(device_id, sample_rate),
            daemon=True
        )
        self.transcription_thread = threading.Thread(
            target=self._transcription_thread,
            args=(sample_rate, lang_code),
            daemon=True
        )

        self.audio_thread.start()
        self.transcription_thread.start()

    def _stop_transcription(self):
        """Stop audio capture and transcription."""
        self.is_recording = False
        self.stop_event.set()

        # Wait for threads to finish (with timeout)
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join(timeout=1.0)

        # Update UI
        self._set_status("Ready", READY_COLOR)
        self.start_button.configure(text="● Iniciar")
        self.model_dropdown.configure(state="normal")
        self.lang_dropdown.configure(state="normal")

    def _audio_capture_thread(self, device_id: int, sample_rate: int):
        """Thread function for audio capture with small blocks for VAD."""
        block_samples = int(sample_rate * AUDIO_BLOCK_DURATION)

        def callback(indata, frames, time_info, status):
            if not self.stop_event.is_set():
                self.audio_queue.put(indata.copy())

        try:
            with sd.InputStream(
                device=device_id,
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=block_samples,
                callback=callback
            ):
                while not self.stop_event.is_set():
                    self.stop_event.wait(timeout=0.1)
        except Exception as e:
            self.text_queue.put(f"[Audio error: {e}]")

    def _transcription_thread(self, sample_rate: int, language: str):
        """Thread function for transcription with VAD-based chunking."""
        audio_buffer = []
        buffer_duration = 0.0
        silence_duration = 0.0

        def compute_rms(audio_chunk: np.ndarray) -> float:
            """Compute RMS energy of audio chunk."""
            return float(np.sqrt(np.mean(audio_chunk ** 2)))

        def should_send() -> bool:
            """Determine if we should send audio to the model."""
            # Force send if buffer exceeds maximum
            if buffer_duration >= MAX_CHUNK_DURATION:
                return True
            # Send if silence detected and buffer has minimum duration
            if silence_duration >= SILENCE_DURATION and buffer_duration >= MIN_CHUNK_DURATION:
                return True
            return False

        def transcribe_buffer():
            """Send buffer to model and get transcription."""
            nonlocal audio_buffer, buffer_duration, silence_duration

            if not audio_buffer:
                return

            audio_data = np.concatenate(audio_buffer).flatten()

            try:
                segments, _ = self.model.transcribe(
                    audio_data,
                    language=language,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=300)
                )

                for segment in segments:
                    text = segment.text.strip()
                    if text:
                        self.text_queue.put(text)
            except Exception as e:
                self.text_queue.put(f"[Transcription error: {e}]")

            # Reset buffer
            audio_buffer = []
            buffer_duration = 0.0
            silence_duration = 0.0

        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.5)
                audio_buffer.append(chunk)
                chunk_duration = len(chunk) / sample_rate
                buffer_duration += chunk_duration

                # Check if this chunk is silence
                rms = compute_rms(chunk)
                if rms < SILENCE_THRESHOLD:
                    silence_duration += chunk_duration
                else:
                    silence_duration = 0.0  # Reset on speech

                # Check if we should send to model
                if should_send():
                    transcribe_buffer()

            except queue.Empty:
                # On timeout, check if we have pending audio with silence
                if buffer_duration >= MIN_CHUNK_DURATION and silence_duration > 0:
                    transcribe_buffer()
                continue

        # Transcribe any remaining audio when stopping
        if audio_buffer and buffer_duration >= MIN_CHUNK_DURATION:
            transcribe_buffer()

    def _update_text_from_queue(self):
        """Periodically check text queue and update text box."""
        try:
            while True:
                text = self.text_queue.get_nowait()
                self.text_box.insert("end", text + " ")
                self.text_box.see("end")
        except queue.Empty:
            pass

        # Schedule next update
        self.after(100, self._update_text_from_queue)

    def _set_status(self, text: str, color: str):
        """Update status indicator."""
        self.status_dot.configure(text_color=color)
        self.status_label.configure(text=text)

    def _copy_to_clipboard(self):
        """Copy text box content to clipboard."""
        text = self.text_box.get("1.0", "end-1c")
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)

            # Visual feedback
            original_text = self.copy_button.cget("text")
            self.copy_button.configure(text="Copiado!")
            self.after(1000, lambda: self.copy_button.configure(text=original_text))

    def _clear_text(self):
        """Clear the text box."""
        self.text_box.delete("1.0", "end")

    def _on_close(self):
        """Handle window close event."""
        if self.is_recording:
            self._stop_transcription()
        self.destroy()


def main():
    """Entry point."""
    app = TranscriptionApp()
    app.mainloop()


if __name__ == '__main__':
    main()
