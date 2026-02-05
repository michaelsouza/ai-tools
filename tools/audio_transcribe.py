#!/usr/bin/env python3
"""
Real-time Audio Transcription using Faster-Whisper

This script provides real-time speech-to-text transcription using the Faster-Whisper
model running locally on your machine. It can transcribe from microphone input in
real-time or process audio files in batch mode.

Usage:
    # Real-time microphone transcription (default)
    python audio_transcribe.py

    # Real-time with specific model
    python audio_transcribe.py -m small

    # Transcribe audio file (batch mode)
    python audio_transcribe.py -i audio.mp3

    # Transcribe with output to file
    python audio_transcribe.py -o transcript.txt

    # Transcribe file with timestamps
    python audio_transcribe.py -i audio.mp3 --timestamps

    # Use GPU acceleration (if available)
    python audio_transcribe.py -d cuda

    # Specify language hint for better accuracy
    python audio_transcribe.py -l pt

    # Customize chunk duration for real-time mode
    python audio_transcribe.py --chunk-duration 5.0

    # Verbose mode with device info
    python audio_transcribe.py -v

Options:
    -i, --input FILE         Audio file to transcribe (batch mode)
    -o, --output FILE        Save transcript to file (default: stdout only)
    -m, --model SIZE         Model size: tiny, base, small, medium, large-v3 (default: base)
    -d, --device DEVICE      Device: cpu, cuda, auto (default: auto)
    -l, --language LANG      Language code hint (e.g., en, pt, es)
    --chunk-duration SEC     Real-time chunk duration in seconds (default: 3.0)
    --compute-type TYPE      Compute type: int8, float16, float32 (default: auto)
    --timestamps             Include timestamps in output
    -v, --verbose            Show detailed processing information

Models:
    tiny     - Fastest, lowest quality (~400MB RAM, ~32x real-time)
    base     - Good balance (default) (~500MB RAM, ~16x real-time)
    small    - Better quality (~1GB RAM, ~6x real-time)
    medium   - High quality (~2.5GB RAM, ~2x real-time)
    large-v3 - Best quality (~5GB RAM, may be slower than real-time on CPU)

Supported Audio Formats:
    WAV, MP3, M4A, FLAC, OGG, WEBM

Requirements:
    - faster-whisper: Local transcription engine
    - sounddevice: Microphone audio capture
    - numpy: Audio buffer handling
    - rich: Terminal formatting

WSL Notes:
    - Audio input requires PulseAudio configured in WSL
    - GPU acceleration requires Windows 11 + NVIDIA drivers with CUDA
    - If no audio devices found, check PulseAudio setup:
      1. Install: sudo apt install pulseaudio
      2. Start: pulseaudio --start
      3. Check: pactl list sources

Exit Codes:
    0: Success
    1: Processing error (no microphone, file not found, transcription failed)
    2: Configuration error (invalid model, missing dependencies)
    130: Interrupted by user (Ctrl+C)
"""

import argparse
import os
import sys
import queue
import threading
from pathlib import Path
from typing import Optional, List, Tuple

# Check for required dependencies
try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Error: faster-whisper not installed.", file=sys.stderr)
    print("Install with: pip install faster-whisper", file=sys.stderr)
    sys.exit(2)

try:
    import sounddevice as sd
except ImportError:
    print("Error: sounddevice not installed.", file=sys.stderr)
    print("Install with: pip install sounddevice", file=sys.stderr)
    sys.exit(2)

try:
    import numpy as np
except ImportError:
    print("Error: numpy not installed.", file=sys.stderr)
    print("Install with: pip install numpy", file=sys.stderr)
    sys.exit(2)

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Constants
SUPPORTED_FORMATS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm'}
MODEL_SIZES = ['tiny', 'base', 'small', 'medium', 'large-v3']
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_DURATION = 3.0


def get_audio_device_info(verbose: bool, console: Console) -> Optional[Tuple[int, int]]:
    """
    Detect default microphone and return (device_id, sample_rate).
    Returns None if no input device available.
    """
    try:
        devices = sd.query_devices()
        default_input = sd.query_devices(kind='input')

        if verbose:
            console.print("\n[bold]Available Audio Devices:[/]")
            for i, device in enumerate(devices):
                marker = " [green](default input)[/]" if i == default_input.get('index', -1) else ""
                if device['max_input_channels'] > 0:
                    console.print(f"  [{i}] {device['name']}{marker}")

        if default_input is None or default_input.get('max_input_channels', 0) == 0:
            return None

        device_id = default_input.get('index', 0)
        # Use 16kHz for Whisper (optimal sample rate)
        return device_id, DEFAULT_SAMPLE_RATE

    except Exception as e:
        if verbose:
            console.print(f"[yellow]Warning: Error querying audio devices: {e}[/]")
        return None


def detect_device_and_compute_type(device: str, compute_type: str, verbose: bool, console: Console) -> Tuple[str, str]:
    """
    Auto-detect device and compute type if set to 'auto'.
    Returns (device, compute_type) tuple.
    """
    # Detect device
    if device == 'auto':
        try:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                if verbose:
                    console.print(f"[green]GPU detected: {torch.cuda.get_device_name(0)}[/]")
            else:
                device = 'cpu'
                if verbose:
                    console.print("[yellow]No GPU detected, using CPU[/]")
        except ImportError:
            device = 'cpu'
            if verbose:
                console.print("[yellow]PyTorch not available, using CPU[/]")

    # Auto-select compute type based on device
    if compute_type == 'auto':
        if device == 'cuda':
            compute_type = 'float16'
        else:
            compute_type = 'int8'
        if verbose:
            console.print(f"[dim]Compute type: {compute_type}[/]")

    return device, compute_type


def load_transcription_model(
    model_size: str,
    device: str,
    compute_type: str,
    console: Console
) -> Optional[WhisperModel]:
    """
    Load and initialize the Faster-Whisper model.
    Downloads model if not cached locally.
    """
    with console.status(f"[bold blue]Loading model '{model_size}' on {device}...", spinner="dots"):
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            return model
        except Exception as e:
            console.print(f"[bold red]Error loading model:[/] {e}")
            if 'CUDA' in str(e) or 'cuda' in str(e):
                console.print("[yellow]Tip: Try using -d cpu if GPU is not available[/]")
            if 'memory' in str(e).lower():
                console.print("[yellow]Tip: Try a smaller model with -m tiny or -m base[/]")
            return None


def format_timestamp(seconds: float) -> str:
    """Format seconds as [MM:SS.s] timestamp."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"[{minutes:02d}:{secs:05.2f}]"


def format_transcript(segments, show_timestamps: bool) -> str:
    """
    Format transcription segments into readable text.
    Optionally includes timestamp markers.
    """
    parts = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if show_timestamps:
            timestamp = format_timestamp(segment.start)
            parts.append(f"{timestamp} {text}")
        else:
            parts.append(text)

    return "\n".join(parts) if show_timestamps else " ".join(parts)


def transcribe_realtime(
    model: WhisperModel,
    model_info: str,
    device_id: int,
    sample_rate: int,
    chunk_duration: float,
    language: Optional[str],
    output_file: Optional[Path],
    show_timestamps: bool,
    console: Console
) -> int:
    """
    Capture audio from microphone and transcribe in real-time.
    Returns exit code.
    """
    audio_queue: queue.Queue = queue.Queue()
    is_running = True
    total_duration = 0.0
    all_text: List[str] = []

    def audio_callback(indata, frames, time_info, status):
        """Callback for audio stream - adds audio chunks to queue."""
        if status:
            console.print(f"[yellow]Audio warning: {status}[/]")
        audio_queue.put(indata.copy())

    # Display info panel
    lang_display = language if language else "auto"
    console.print(Panel(
        f"[bold]Real-time Audio Transcription[/]\n"
        f"Model: [cyan]{model_info}[/] | "
        f"Language: [cyan]{lang_display}[/]",
        border_style="blue",
        title="Configuration"
    ))
    console.print("\n[bold green]Listening...[/] (Press Ctrl+C to stop)\n")

    try:
        # Calculate chunk size in samples
        chunk_samples = int(sample_rate * chunk_duration)

        # Start audio stream
        with sd.InputStream(
            device=device_id,
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=chunk_samples,
            callback=audio_callback
        ):
            audio_buffer = []
            buffer_duration = 0.0

            while is_running:
                try:
                    # Get audio chunk from queue (with timeout to allow Ctrl+C)
                    chunk = audio_queue.get(timeout=0.5)
                    audio_buffer.append(chunk)
                    buffer_duration += len(chunk) / sample_rate

                    # Process when we have enough audio
                    if buffer_duration >= chunk_duration:
                        # Combine buffer
                        audio_data = np.concatenate(audio_buffer).flatten()

                        # Transcribe
                        segments, info = model.transcribe(
                            audio_data,
                            language=language,
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=500)
                        )

                        # Process segments
                        for segment in segments:
                            text = segment.text.strip()
                            if text:
                                if show_timestamps:
                                    timestamp = format_timestamp(total_duration + segment.start)
                                    line = f"{timestamp} {text}"
                                    console.print(f"[dim]{timestamp}[/] {text}")
                                else:
                                    line = text
                                    console.print(text)
                                all_text.append(line)

                        total_duration += buffer_duration
                        audio_buffer = []
                        buffer_duration = 0.0

                except queue.Empty:
                    continue

    except KeyboardInterrupt:
        is_running = False
        console.print("\n\n[yellow]Processing interrupted by user.[/]")

    except Exception as e:
        console.print(f"\n[bold red]Error during transcription:[/] {e}")
        return 1

    # Save output if requested
    if output_file and all_text:
        try:
            separator = "\n" if show_timestamps else " "
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(separator.join(all_text), encoding='utf-8')
            console.print(Panel(
                f"[bold green]Transcription Complete[/]\n"
                f"Duration: [cyan]{total_duration:.1f}s[/]\n"
                f"Output saved to: [cyan]{output_file}[/]",
                border_style="green",
                title="Done"
            ))
        except Exception as e:
            console.print(f"[bold red]Error saving output:[/] {e}")
            return 1
    elif all_text:
        console.print(Panel(
            f"[bold green]Transcription Complete[/]\n"
            f"Duration: [cyan]{total_duration:.1f}s[/]",
            border_style="green",
            title="Done"
        ))
    else:
        console.print("[yellow]No speech detected.[/]")

    return 130 if not all_text or is_running == False else 0


def transcribe_file(
    model: WhisperModel,
    model_info: str,
    audio_path: Path,
    language: Optional[str],
    output_file: Optional[Path],
    show_timestamps: bool,
    console: Console
) -> int:
    """
    Transcribe an audio file and display/save results.
    Returns exit code.
    """
    # Validate file
    if not audio_path.exists():
        console.print(f"[bold red]Error:[/] File not found: {audio_path}")
        return 1

    if audio_path.suffix.lower() not in SUPPORTED_FORMATS:
        console.print(f"[bold red]Error:[/] Unsupported format: {audio_path.suffix}")
        console.print(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        return 2

    # Display info panel
    lang_display = language if language else "auto"
    console.print(Panel(
        f"[bold]Audio File Transcription[/]\n"
        f"File: [cyan]{audio_path.name}[/]\n"
        f"Model: [cyan]{model_info}[/] | "
        f"Language: [cyan]{lang_display}[/]",
        border_style="blue",
        title="Configuration"
    ))

    # Transcribe with progress
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("[bold blue]Transcribing...", total=100)

            # Transcribe file
            segments, info = model.transcribe(
                str(audio_path),
                language=language,
                vad_filter=True
            )

            # Collect segments (generator)
            all_segments = []
            for segment in segments:
                all_segments.append(segment)
                # Estimate progress based on segment timing
                if info.duration > 0:
                    pct = min(100, (segment.end / info.duration) * 100)
                    progress.update(task, completed=pct)

            progress.update(task, completed=100)

        # Format transcript
        transcript = format_transcript(all_segments, show_timestamps)

        if not transcript.strip():
            console.print("[yellow]No speech detected in audio file.[/]")
            return 0

        # Save or display
        if output_file:
            try:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_text(transcript, encoding='utf-8')
                console.print(Panel(
                    f"[bold green]Transcription Complete[/]\n"
                    f"Duration: [cyan]{info.duration:.1f}s[/]\n"
                    f"Output saved to: [cyan]{output_file}[/]",
                    border_style="green",
                    title="Done"
                ))
            except Exception as e:
                console.print(f"[bold red]Error saving output:[/] {e}")
                return 1
        else:
            console.print(Panel(
                f"[bold green]Transcription Complete[/]\n"
                f"Duration: [cyan]{info.duration:.1f}s[/]",
                border_style="green",
                title="Done"
            ))
            console.print("\n[bold]Transcript:[/]")
            console.print(transcript)

        return 0

    except Exception as e:
        console.print(f"[bold red]Error during transcription:[/] {e}")
        return 1


def main() -> int:
    """Main function orchestrating the transcription workflow."""
    console = Console()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Real-time audio transcription using Faster-Whisper (local)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Real-time mic transcription
  %(prog)s -m small -l pt            # Better quality, Portuguese
  %(prog)s -i audio.mp3              # Transcribe file
  %(prog)s -i audio.mp3 -o out.txt   # Transcribe file to output
  %(prog)s --timestamps              # Include timestamps
  %(prog)s -d cuda -m medium         # GPU with medium model
        """
    )
    parser.add_argument('-i', '--input', dest='input_file', type=str,
                        help='Audio file to transcribe (batch mode)')
    parser.add_argument('-o', '--output', dest='output_file', type=str,
                        help='Save transcript to file')
    parser.add_argument('-m', '--model', dest='model_size', type=str, default='base',
                        choices=MODEL_SIZES, help='Model size (default: base)')
    parser.add_argument('-d', '--device', type=str, default='auto',
                        choices=['cpu', 'cuda', 'auto'], help='Device (default: auto)')
    parser.add_argument('-l', '--language', type=str, default=None,
                        help='Language code hint (e.g., en, pt, es)')
    parser.add_argument('--chunk-duration', type=float, default=DEFAULT_CHUNK_DURATION,
                        help=f'Real-time chunk duration in seconds (default: {DEFAULT_CHUNK_DURATION})')
    parser.add_argument('--compute-type', type=str, default='auto',
                        choices=['auto', 'int8', 'float16', 'float32'],
                        help='Compute type (default: auto)')
    parser.add_argument('--timestamps', action='store_true',
                        help='Include timestamps in output')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show detailed processing information')

    args = parser.parse_args()

    # Validate model
    if args.model_size not in MODEL_SIZES:
        console.print(f"[bold red]Error:[/] Invalid model: {args.model_size}")
        console.print(f"Valid models: {', '.join(MODEL_SIZES)}")
        return 2

    # Detect device and compute type
    device, compute_type = detect_device_and_compute_type(
        args.device, args.compute_type, args.verbose, console
    )

    # Check for real-time mode requirements
    if not args.input_file:
        audio_info = get_audio_device_info(args.verbose, console)
        if audio_info is None:
            console.print("[bold red]Error:[/] No microphone detected.")
            console.print("\n[yellow]Suggestions:[/]")
            console.print("  - Connect a microphone")
            console.print("  - Use -i to transcribe an audio file instead")
            console.print("  - In WSL, ensure PulseAudio is configured:")
            console.print("    1. sudo apt install pulseaudio")
            console.print("    2. pulseaudio --start")
            return 1
        device_id, sample_rate = audio_info

    # Load model
    model = load_transcription_model(args.model_size, device, compute_type, console)
    if model is None:
        return 2

    # Prepare output file path
    output_path = Path(args.output_file).expanduser().resolve() if args.output_file else None

    # Model info string for display
    model_info = f"{args.model_size} ({device}, {compute_type})"

    # Run transcription
    try:
        if args.input_file:
            # Batch mode - transcribe file
            input_path = Path(args.input_file).expanduser().resolve()
            return transcribe_file(
                model=model,
                model_info=model_info,
                audio_path=input_path,
                language=args.language,
                output_file=output_path,
                show_timestamps=args.timestamps,
                console=console
            )
        else:
            # Real-time mode - transcribe from microphone
            return transcribe_realtime(
                model=model,
                model_info=model_info,
                device_id=device_id,
                sample_rate=sample_rate,
                chunk_duration=args.chunk_duration,
                language=args.language,
                output_file=output_path,
                show_timestamps=args.timestamps,
                console=console
            )
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user.[/]")
        return 130
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/] {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
