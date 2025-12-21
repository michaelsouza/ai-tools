#!/home/michael/.venv/bin/python
"""
YouTube to Audio Downloader

This script downloads audio from YouTube videos and playlists in various formats.
It uses yt-dlp to extract the best quality audio and convert it to your preferred format.

Usage:
    # Basic usage - download as MP3
    python youtube2audio.py "https://www.youtube.com/watch?v=VIDEO_ID"

    # Download to specific directory
    python youtube2audio.py "URL" -o ~/Music

    # Choose audio format
    python youtube2audio.py "URL" -f m4a

    # Set audio quality (0=best, 9=worst for VBR formats)
    python youtube2audio.py "URL" -q 0

    # Custom filename template
    python youtube2audio.py "URL" --template "%(artist)s - %(title)s"

    # Download playlist
    python youtube2audio.py "https://www.youtube.com/playlist?list=..." -o ~/Music/Playlist

Features:
    - Supports YouTube, YouTube Music, and other yt-dlp compatible sites
    - Multiple audio formats: MP3, M4A, OPUS, FLAC, WAV
    - Automatic metadata embedding (title, artist, album, etc.)
    - Progress bar showing download status
    - Playlist support
    - Resume interrupted downloads

Requirements:
    - yt-dlp: Modern YouTube downloader
    - ffmpeg: Audio conversion (must be installed on system)

Note:
    ffmpeg must be installed separately:
        Ubuntu/Debian: sudo apt install ffmpeg
        Arch: sudo pacman -S ffmpeg
        macOS: brew install ffmpeg
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    import yt_dlp
except ImportError:
    print("Error: yt-dlp is not installed.", file=sys.stderr)
    print("Install it with: pip install yt-dlp", file=sys.stderr)
    sys.exit(1)


# Supported audio formats and their properties
AUDIO_FORMATS = {
    'mp3': {'ext': 'mp3', 'quality_param': True},
    'm4a': {'ext': 'm4a', 'quality_param': True},
    'opus': {'ext': 'opus', 'quality_param': True},
    'flac': {'ext': 'flac', 'quality_param': False},
    'wav': {'ext': 'wav', 'quality_param': False},
}


def download_audio(
    url: str,
    output_dir: Optional[str] = None,
    audio_format: str = 'mp3',
    quality: int = 0,
    filename_template: Optional[str] = None,
    verbose: bool = False
) -> None:
    """
    Download audio from YouTube URL.

    Args:
        url: YouTube video or playlist URL
        output_dir: Directory to save audio files (default: current directory)
        audio_format: Output audio format (mp3, m4a, opus, flac, wav)
        quality: Audio quality 0-9 (0=best, 9=worst) for VBR formats
        filename_template: Custom filename template
        verbose: Print detailed progress information
    """
    # Validate format
    if audio_format not in AUDIO_FORMATS:
        raise ValueError(f"Unsupported format: {audio_format}. Choose from: {', '.join(AUDIO_FORMATS.keys())}")

    # Setup output path
    if output_dir:
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path.cwd()

    # Default filename template
    if filename_template is None:
        filename_template = '%(title)s.%(ext)s'
    elif not filename_template.endswith('.%(ext)s'):
        filename_template += '.%(ext)s'

    # Full output template
    output_template = str(output_path / filename_template)

    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'quiet': not verbose,
        'no_warnings': not verbose,
        'extract_flat': False,
        # Network settings
        'socket_timeout': 60,
        'retries': 10,
        'fragment_retries': 10,
        'http_chunk_size': 10485760,  # 10MB chunks
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': audio_format,
        }],
        # Embed metadata
        'writethumbnail': True,
        'postprocessor_args': [
            '-metadata', 'title=%(title)s',
            '-metadata', 'artist=%(artist)s',
            '-metadata', 'album=%(album)s',
        ],
        'add_metadata': True,
    }

    # Add quality setting if format supports it
    if AUDIO_FORMATS[audio_format]['quality_param']:
        ydl_opts['postprocessors'][0]['preferredquality'] = str(quality)

    # Embed thumbnail as cover art for m4a/mp3
    if audio_format in ['m4a', 'mp3']:
        ydl_opts['postprocessors'].append({
            'key': 'EmbedThumbnail',
        })

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info first
            if verbose:
                print(f"Fetching info for: {url}")

            info = ydl.extract_info(url, download=False)

            # Check if it's a playlist
            if 'entries' in info:
                print(f"Downloading playlist: {info.get('title', 'Unknown')}")
                print(f"Total videos: {len(info['entries'])}")
            else:
                print(f"Downloading: {info.get('title', 'Unknown')}")

            # Download
            ydl.download([url])

            print(f"\n✓ Download complete! Files saved to: {output_path}")

    except yt_dlp.utils.DownloadError as e:
        print(f"Error downloading: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDownload cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download audio from YouTube videos and playlists",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download as MP3 (default)
  %(prog)s "https://www.youtube.com/watch?v=VIDEO_ID"

  # Download to specific folder
  %(prog)s "URL" -o ~/Music/Downloads

  # Download as high-quality M4A
  %(prog)s "URL" -f m4a -q 0

  # Download with custom filename
  %(prog)s "URL" --template "%%(artist)s - %%(title)s"

  # Download entire playlist
  %(prog)s "https://www.youtube.com/playlist?list=..." -o ~/Music/MyPlaylist

Supported formats: mp3, m4a, opus, flac, wav
Quality range: 0 (best) to 9 (worst) - only for mp3, m4a, opus
        """
    )

    parser.add_argument(
        'url',
        help='YouTube video or playlist URL'
    )

    parser.add_argument(
        '-o', '--output',
        dest='output_dir',
        help='Output directory (default: current directory)'
    )

    parser.add_argument(
        '-f', '--format',
        dest='audio_format',
        choices=list(AUDIO_FORMATS.keys()),
        default='mp3',
        help='Audio format (default: mp3)'
    )

    parser.add_argument(
        '-q', '--quality',
        type=int,
        choices=range(10),
        default=0,
        help='Audio quality 0-9, where 0=best (default: 0) - only for VBR formats'
    )

    parser.add_argument(
        '-t', '--template',
        dest='filename_template',
        help='Filename template (default: "%%(title)s") - see yt-dlp output template for options'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed progress information'
    )

    args = parser.parse_args()

    # Run download
    download_audio(
        url=args.url,
        output_dir=args.output_dir,
        audio_format=args.audio_format,
        quality=args.quality,
        filename_template=args.filename_template,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
