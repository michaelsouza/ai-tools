#!/home/michael/.venv/bin/python
"""
Demucs Music Source Separation

This script separates music into individual stems (vocals, drums, bass, other instruments)
using the Hybrid Demucs model from Meta/Facebook Research.

Usage:
    # Separate a single audio file
    python demucs_separate.py audio.mp3

    # Separate multiple files
    python demucs_separate.py song1.mp3 song2.wav song3.flac

    # Specify output directory
    python demucs_separate.py audio.mp3 -o output_folder

    # Separate only vocals from the rest
    python demucs_separate.py audio.mp3 --two-stems vocals

    # Use different model
    python demucs_separate.py audio.mp3 -m htdemucs_ft

    # Output as MP3 with custom bitrate
    python demucs_separate.py audio.mp3 --mp3 --mp3-bitrate 320

    # Output as WAV (float32)
    python demucs_separate.py audio.mp3 --float32

Features:
    - Separates music into: vocals, drums, bass, other
    - Multiple model options (htdemucs, htdemucs_ft, mdx_extra, etc.)
    - Output formats: MP3, WAV (float32), WAV (int24)
    - Two-stems mode for quick vocal/instrumental separation
    - Batch processing of multiple files
    - Progress tracking

Models:
    - htdemucs: Default model, good balance of quality and speed
    - htdemucs_ft: Fine-tuned version, better quality
    - htdemucs_6s: Separates into 6 stems (adds piano and guitar)
    - mdx_extra: Extra quality model (slower)
    - mdx_extra_q: Quantized version (faster)

Requirements:
    - demucs: pip install demucs
    - ffmpeg: System package (sudo apt install ffmpeg)
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Fix torchaudio backend to avoid torchcodec dependency
# Set environment variable before torchaudio is imported by demucs
import os
os.environ['TORCHAUDIO_USE_BACKEND_DISPATCHER'] = '0'


# Supported audio extensions
AUDIO_EXTENSIONS = ['mp3', 'wav', 'ogg', 'flac', 'm4a', 'aac', 'wma']

# Available models
MODELS = [
    'htdemucs',
    'htdemucs_ft',
    'htdemucs_6s',
    'mdx',
    'mdx_extra',
    'mdx_extra_q',
]

# Two-stems options
TWO_STEMS_OPTIONS = ['vocals', 'drums', 'bass', 'other']


def check_demucs_installed() -> bool:
    """Check if demucs is installed."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'demucs.separate', '--help'],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def find_audio_files(paths: List[str]) -> List[Path]:
    """
    Find all valid audio files from given paths.

    Args:
        paths: List of file paths or directory paths

    Returns:
        List of Path objects for valid audio files
    """
    audio_files = []

    for path_str in paths:
        path = Path(path_str).expanduser().resolve()

        if not path.exists():
            print(f"Warning: {path} does not exist, skipping...", file=sys.stderr)
            continue

        if path.is_file():
            if path.suffix.lower().lstrip('.') in AUDIO_EXTENSIONS:
                audio_files.append(path)
            else:
                print(f"Warning: {path} is not a supported audio file, skipping...", file=sys.stderr)

        elif path.is_dir():
            # Find all audio files in directory
            for ext in AUDIO_EXTENSIONS:
                audio_files.extend(path.glob(f'*.{ext}'))
                audio_files.extend(path.glob(f'*.{ext.upper()}'))

    return sorted(set(audio_files))


def separate_audio(
    audio_files: List[Path],
    output_dir: Optional[str] = None,
    model: str = 'htdemucs',
    two_stems: Optional[str] = None,
    mp3: bool = False,
    mp3_bitrate: int = 320,
    float32: bool = False,
    int24: bool = False,
    device: str = 'cpu',
    jobs: int = 0,
    verbose: bool = False
) -> int:
    """
    Separate audio files using Demucs.

    Args:
        audio_files: List of audio file paths to process
        output_dir: Output directory (default: ./separated)
        model: Demucs model to use
        two_stems: Separate only one stem from the rest
        mp3: Output as MP3 files
        mp3_bitrate: MP3 bitrate (64-320)
        float32: Output as float32 WAV files
        int24: Output as int24 WAV files
        device: Device to use (cpu, cuda, mps)
        jobs: Number of parallel jobs (0=auto)
        verbose: Show detailed output

    Returns:
        Exit code (0=success, 1=error)
    """
    # Setup output directory
    if output_dir:
        out_path = Path(output_dir).expanduser().resolve()
    else:
        out_path = Path.cwd() / 'separated'

    out_path.mkdir(parents=True, exist_ok=True)

    # Build demucs command
    cmd = [
        sys.executable,
        '-m', 'demucs.separate',
        '-o', str(out_path),
        '-n', model,
        '-d', device,
    ]

    if jobs > 0:
        cmd += ['-j', str(jobs)]

    # Output format options
    if mp3:
        cmd += ['--mp3', f'--mp3-bitrate={mp3_bitrate}']
    elif float32:
        cmd += ['--float32']
    elif int24:
        cmd += ['--int24']

    # Two stems mode
    if two_stems:
        cmd += [f'--two-stems={two_stems}']

    # Add audio files
    cmd += [str(f) for f in audio_files]

    # Print info
    print(f"Processing {len(audio_files)} file(s)...")
    print(f"Model: {model}")
    print(f"Output directory: {out_path}")
    if two_stems:
        print(f"Mode: Two-stems ({two_stems} vs rest)")
    else:
        print(f"Mode: Full separation (vocals, drums, bass, other)")
    print()

    if verbose:
        print("Command:", ' '.join(cmd))
        print()

    # Run demucs
    try:
        result = subprocess.run(
            cmd,
            check=False,
            stdout=None if verbose else subprocess.PIPE,
            stderr=None if verbose else subprocess.STDOUT,
        )

        if result.returncode == 0:
            print(f"\n✓ Separation complete!")
            print(f"  Output saved to: {out_path}")

            # Show output structure
            if two_stems:
                print(f"  Files: {two_stems}/ and no_{two_stems}/")
            else:
                print(f"  Stems: vocals/, drums/, bass/, other/")

            return 0
        else:
            print(f"\n✗ Separation failed with exit code {result.returncode}", file=sys.stderr)
            if not verbose and result.stdout:
                print(result.stdout.decode(), file=sys.stderr)
            return 1

    except KeyboardInterrupt:
        print("\n\nSeparation cancelled by user.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(
        description='Separate music into individual stems using Demucs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - separate single file
  %(prog)s song.mp3

  # Separate multiple files
  %(prog)s song1.mp3 song2.wav song3.flac

  # Process entire directory
  %(prog)s /path/to/music/folder

  # Separate only vocals
  %(prog)s song.mp3 --two-stems vocals

  # Use fine-tuned model with MP3 output
  %(prog)s song.mp3 -m htdemucs_ft --mp3 --mp3-bitrate 320

  # 6-stem separation (vocals, drums, bass, other, piano, guitar)
  %(prog)s song.mp3 -m htdemucs_6s

Models:
  htdemucs       - Default, good quality and speed (4 stems)
  htdemucs_ft    - Fine-tuned, better quality (4 stems)
  htdemucs_6s    - 6 stems: vocals, drums, bass, other, piano, guitar
  mdx_extra      - Extra quality (slower)
  mdx_extra_q    - Quantized version (faster)

Output:
  By default, creates a 'separated/' folder with subfolders for each stem.
  Use -o to specify a different output directory.
        """
    )

    parser.add_argument(
        'inputs',
        nargs='+',
        help='Audio file(s) or directory to process'
    )

    parser.add_argument(
        '-o', '--output',
        dest='output_dir',
        help='Output directory (default: ./separated)'
    )

    parser.add_argument(
        '-m', '--model',
        choices=MODELS,
        default='htdemucs',
        help='Demucs model to use (default: htdemucs)'
    )

    parser.add_argument(
        '--two-stems',
        choices=TWO_STEMS_OPTIONS,
        help='Separate only one stem from the rest (faster)'
    )

    parser.add_argument(
        '--mp3',
        action='store_true',
        help='Output as MP3 files instead of WAV'
    )

    parser.add_argument(
        '--mp3-bitrate',
        type=int,
        choices=[64, 96, 128, 160, 192, 256, 320],
        default=320,
        help='MP3 bitrate in kbps (default: 320)'
    )

    parser.add_argument(
        '--float32',
        action='store_true',
        help='Output as float32 WAV files (high quality)'
    )

    parser.add_argument(
        '--int24',
        action='store_true',
        help='Output as int24 WAV files'
    )

    parser.add_argument(
        '-d', '--device',
        choices=['cpu', 'cuda', 'mps'],
        default='cpu',
        help='Device to use for processing (default: cpu)'
    )

    parser.add_argument(
        '-j', '--jobs',
        type=int,
        default=0,
        help='Number of parallel jobs (default: auto)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed output'
    )

    args = parser.parse_args()

    # Check if demucs is installed
    if not check_demucs_installed():
        print("Error: Demucs is not installed.", file=sys.stderr)
        print("Install it with: pip install demucs", file=sys.stderr)
        sys.exit(1)

    # Validate output format options
    if sum([args.mp3, args.float32, args.int24]) > 1:
        print("Error: Can only specify one output format (--mp3, --float32, or --int24)", file=sys.stderr)
        sys.exit(1)

    # Find audio files
    audio_files = find_audio_files(args.inputs)

    if not audio_files:
        print("Error: No valid audio files found.", file=sys.stderr)
        sys.exit(1)

    # Run separation
    exit_code = separate_audio(
        audio_files=audio_files,
        output_dir=args.output_dir,
        model=args.model,
        two_stems=args.two_stems,
        mp3=args.mp3,
        mp3_bitrate=args.mp3_bitrate,
        float32=args.float32,
        int24=args.int24,
        device=args.device,
        jobs=args.jobs,
        verbose=args.verbose
    )

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
