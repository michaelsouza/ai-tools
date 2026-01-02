#!/usr/bin/env python3
"""
Convert HTML files to high-resolution PNG images with automatic cropping.

This tool renders HTML files using a headless browser, captures screenshots
at high DPI, and optionally crops whitespace around the content.

Usage:
    python html2png.py input.html [output.png] [options]

Examples:
    python html2png.py diagram.html                     # Output: diagram.png
    python html2png.py diagram.html output.png          # Specify output
    python html2png.py diagram.html -s 3                # 3x scale (default)
    python html2png.py diagram.html --no-crop           # Skip auto-crop
    python html2png.py diagram.html -p 20               # 20px padding after crop
    python html2png.py diagram.html --selector "#svg"   # Capture specific element
"""

import argparse
import asyncio
import sys
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Error: Required packages not found. Install with:")
    print("  pip install Pillow numpy")
    sys.exit(1)


def parse_color(color_str: str) -> tuple:
    """Parse color string to RGB tuple."""
    color_str = color_str.strip()

    # Handle hex colors
    if color_str.startswith('#'):
        hex_color = color_str[1:]
        if len(hex_color) == 3:
            hex_color = ''.join(c * 2 for c in hex_color)
        if len(hex_color) == 6:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    # Handle named colors
    named_colors = {
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
    }
    if color_str.lower() in named_colors:
        return named_colors[color_str.lower()]

    # Handle RGB tuple string like "255,255,255"
    if ',' in color_str:
        parts = [int(p.strip()) for p in color_str.split(',')]
        if len(parts) >= 3:
            return tuple(parts[:3])

    raise ValueError(f"Cannot parse color: {color_str}")


def find_bounding_box(image: Image.Image, background: tuple = (255, 255, 255),
                      threshold: int = 10) -> tuple:
    """
    Find the bounding box of non-background pixels.

    Args:
        image: PIL Image (RGB or RGBA)
        background: Background color as RGB tuple
        threshold: Maximum distance from background color to consider as background

    Returns:
        Tuple (left, top, right, bottom) or None if image is all background
    """
    img_array = np.array(image)

    # Handle RGBA images
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        alpha = img_array[:, :, 3]
        rgb = img_array[:, :, :3]
    else:
        alpha = np.full(img_array.shape[:2], 255, dtype=np.uint8)
        rgb = img_array if len(img_array.shape) == 3 else np.stack([img_array]*3, axis=-1)

    # Calculate distance from background color
    bg = np.array(background, dtype=np.float32)
    diff = np.sqrt(np.sum((rgb.astype(np.float32) - bg) ** 2, axis=2))

    # Create mask of non-background pixels
    is_content = (alpha > 128) & (diff > threshold)

    # Find bounding box
    rows = np.any(is_content, axis=1)
    cols = np.any(is_content, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]

    return (left, top, right + 1, bottom + 1)


def crop_image(image: Image.Image, padding: int = 0,
               background: tuple = (255, 255, 255), threshold: int = 10) -> Image.Image:
    """
    Crop image by removing background pixels.

    Args:
        image: PIL Image to crop
        padding: Padding to add around content (pixels)
        background: Background color as RGB tuple
        threshold: Threshold for background detection

    Returns:
        Cropped PIL Image
    """
    # Convert to RGB/RGBA for processing
    if image.mode == 'P':
        image = image.convert('RGBA')
    elif image.mode not in ('RGB', 'RGBA'):
        image = image.convert('RGB')

    # Find bounding box
    bbox = find_bounding_box(image, background, threshold)

    if bbox is None:
        return image

    # Add padding
    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image.width, right + padding)
    bottom = min(image.height, bottom + padding)

    return image.crop((left, top, right, bottom))


async def render_html_to_png(html_path: str, output_path: str, scale: int = 3,
                              selector: str = None, viewport_width: int = 1800,
                              viewport_height: int = 1200, wait_ms: int = 500) -> Image.Image:
    """
    Render HTML file to PNG using Playwright.

    Args:
        html_path: Path to HTML file
        output_path: Path to save PNG
        scale: Device scale factor (for high DPI)
        selector: CSS selector for element to capture (captures full page if None)
        viewport_width: Browser viewport width
        viewport_height: Browser viewport height
        wait_ms: Milliseconds to wait for rendering

    Returns:
        PIL Image of the rendered page/element
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("Error: Playwright not found. Install with:")
        print("  pip install playwright")
        print("  playwright install chromium")
        sys.exit(1)

    html_path = Path(html_path).absolute()

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(
            viewport={'width': viewport_width, 'height': viewport_height},
            device_scale_factor=scale
        )

        await page.goto(f"file://{html_path}")
        await page.wait_for_timeout(wait_ms)

        # Determine what to capture
        if selector:
            element = page.locator(selector)
            box = await element.bounding_box()
            if box:
                padding = 5
                clip = {
                    'x': max(0, box['x'] - padding),
                    'y': max(0, box['y'] - padding),
                    'width': box['width'] + 2 * padding,
                    'height': box['height'] + 2 * padding
                }
                screenshot_bytes = await page.screenshot(clip=clip, type='png')
            else:
                screenshot_bytes = await page.screenshot(type='png', full_page=True)
        else:
            screenshot_bytes = await page.screenshot(type='png', full_page=True)

        await browser.close()

    # Convert to PIL Image
    from io import BytesIO
    return Image.open(BytesIO(screenshot_bytes))


def html2png(html_path: str, output_path: str = None, scale: int = 3,
             crop: bool = True, padding: int = 10, selector: str = None,
             background: str = 'white', threshold: int = 10,
             quiet: bool = False) -> dict:
    """
    Convert HTML to PNG with optional cropping.

    Args:
        html_path: Path to input HTML file
        output_path: Path to output PNG (default: same name as input with .png)
        scale: Device scale factor for high DPI rendering
        crop: Whether to auto-crop whitespace
        padding: Padding around content after cropping
        selector: CSS selector for specific element to capture
        background: Background color for crop detection
        threshold: Color threshold for background detection
        quiet: Suppress output messages

    Returns:
        Dictionary with conversion information
    """
    html_path = Path(html_path)
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    if output_path is None:
        output_path = html_path.with_suffix('.png')
    output_path = Path(output_path)

    # Parse background color
    bg_color = parse_color(background)

    # Render HTML
    if not quiet:
        print(f"Rendering: {html_path}")

    image = asyncio.run(render_html_to_png(
        str(html_path),
        str(output_path),
        scale=scale,
        selector=selector
    ))

    original_size = image.size

    # Crop if requested
    if crop:
        image = crop_image(image, padding=padding, background=bg_color, threshold=threshold)

    cropped_size = image.size

    # Save
    image.save(output_path, optimize=True)

    result = {
        'input': str(html_path),
        'output': str(output_path),
        'scale': scale,
        'original_size': original_size,
        'final_size': cropped_size,
        'cropped': crop and (original_size != cropped_size)
    }

    if not quiet:
        print(f"Output: {output_path}")
        print(f"Size: {cropped_size[0]}x{cropped_size[1]} pixels")
        if result['cropped']:
            orig_pixels = original_size[0] * original_size[1]
            final_pixels = cropped_size[0] * cropped_size[1]
            saved = orig_pixels - final_pixels
            print(f"Cropped: {original_size[0]}x{original_size[1]} -> {cropped_size[0]}x{cropped_size[1]} ({100*saved/orig_pixels:.1f}% removed)")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Convert HTML files to high-resolution PNG images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('input', help='Input HTML file path')
    parser.add_argument('output', nargs='?', help='Output PNG file path (default: input with .png extension)')
    parser.add_argument('-s', '--scale', type=int, default=3,
                        help='Device scale factor for high DPI (default: 3)')
    parser.add_argument('--no-crop', action='store_true',
                        help='Disable automatic whitespace cropping')
    parser.add_argument('-p', '--padding', type=int, default=10,
                        help='Padding around content after cropping (default: 10)')
    parser.add_argument('--selector', type=str,
                        help='CSS selector for specific element to capture')
    parser.add_argument('-b', '--background', type=str, default='white',
                        help='Background color for crop detection (default: white)')
    parser.add_argument('-t', '--threshold', type=int, default=10,
                        help='Color threshold for background detection (default: 10)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress output messages')

    args = parser.parse_args()

    try:
        html2png(
            args.input,
            args.output,
            scale=args.scale,
            crop=not args.no_crop,
            padding=args.padding,
            selector=args.selector,
            background=args.background,
            threshold=args.threshold,
            quiet=args.quiet
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
