import asyncio
import os
from pathlib import Path
from playwright.async_api import async_playwright
from PIL import Image
import io

async def export_slides_to_pdf(html_path, output_pdf_path, num_slides=13):
    html_abs_path = os.path.abspath(html_path)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        # Set a fixed viewport and higher device_scale_factor for high-res screenshots
        page = await browser.new_page(
            viewport={'width': 1280, 'height': 800},
            device_scale_factor=2
        )
        
        await page.goto(f"file://{html_abs_path}")
        # Wait for any initial animations or fonts
        await page.wait_for_timeout(1000)
        
        images = []
        
        for i in range(num_slides):
            print(f"Capturing slide {i+1}/{num_slides}...")
            
            # Capture the slide viewer element
            # We use a slight delay to allow animations to finish
            await page.wait_for_timeout(600)
            
            # Selector for the main slide box
            viewer = page.locator("#slide-viewer")
            screenshot_bytes = await viewer.screenshot()
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(screenshot_bytes))
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            images.append(img)
            
            # Move to next slide
            if i < num_slides - 1:
                await page.keyboard.press("ArrowRight")
        
        await browser.close()
        
        if images:
            print(f"Saving {len(images)} slides to {output_pdf_path}...")
            images[0].save(
                output_pdf_path,
                save_all=True,
                append_images=images[1:],
                resolution=200.0,
                quality=100
            )
            print("Export complete!")

if __name__ == "__main__":
    input_html = "ppc/slides.html"
    output_pdf = "ppc/slides.pdf"
    asyncio.run(export_slides_to_pdf(input_html, output_pdf, 14))
