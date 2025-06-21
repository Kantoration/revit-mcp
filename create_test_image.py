#!/usr/bin/env python3
"""
Create a simple test image for visual context analysis.
This script generates a basic image that can be used to test the CLIP integration.
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_test_wall_image():
    """Create a simple test image showing a wall with space for a window."""
    
    # Create a new image with white background
    width, height = 800, 600
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Draw a wall (gray rectangle)
    wall_left = 100
    wall_top = 100
    wall_width = 600
    wall_height = 400
    draw.rectangle([wall_left, wall_top, wall_left + wall_width, wall_top + wall_height], 
                   fill='lightgray', outline='darkgray', width=3)
    
    # Draw a window opening (white rectangle with border)
    window_left = wall_left + 200
    window_top = wall_top + 100
    window_width = 200
    window_height = 150
    draw.rectangle([window_left, window_top, window_left + window_width, window_top + window_height], 
                   fill='white', outline='black', width=2)
    
    # Add text labels
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None
    
    # Add title
    title = "Test Wall with Window Opening"
    if font:
        draw.text((width//2 - 150, 30), title, fill='black', font=font)
    else:
        draw.text((width//2 - 150, 30), title, fill='black')
    
    # Add labels
    labels = [
        ("Wall", wall_left + wall_width//2, wall_top - 20),
        ("Window Opening", window_left + window_width//2, window_top - 20),
        ("Height: 400", wall_left - 50, wall_top + wall_height//2),
        ("Width: 600", wall_left + wall_width//2, wall_top + wall_height + 20)
    ]
    
    for text, x, y in labels:
        if font:
            draw.text((x - 50, y), text, fill='black', font=font)
        else:
            draw.text((x - 50, y), text, fill='black')
    
    # Save the image
    filename = "test_wall.jpg"
    image.save(filename, "JPEG", quality=95)
    print(f"✅ Test image created: {filename}")
    print(f"📏 Image size: {width}x{height} pixels")
    print(f"📁 Saved as: {os.path.abspath(filename)}")
    
    return filename

def create_test_door_image():
    """Create a simple test image showing a wall with a door."""
    
    # Create a new image with white background
    width, height = 800, 600
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Draw a wall (gray rectangle)
    wall_left = 100
    wall_top = 100
    wall_width = 600
    wall_height = 400
    draw.rectangle([wall_left, wall_top, wall_left + wall_width, wall_top + wall_height], 
                   fill='lightgray', outline='darkgray', width=3)
    
    # Draw a door opening (white rectangle with border)
    door_left = wall_left + 250
    door_top = wall_top + 150
    door_width = 100
    door_height = 250
    draw.rectangle([door_left, door_top, door_left + door_width, door_top + door_height], 
                   fill='white', outline='black', width=2)
    
    # Add door frame details
    frame_width = 10
    draw.rectangle([door_left - frame_width, door_top - frame_width, 
                   door_left + door_width + frame_width, door_top + door_height + frame_width], 
                   outline='brown', width=2)
    
    # Add text labels
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    # Add title
    title = "Test Wall with Door Opening"
    if font:
        draw.text((width//2 - 150, 30), title, fill='black', font=font)
    else:
        draw.text((width//2 - 150, 30), title, fill='black')
    
    # Add labels
    labels = [
        ("Wall", wall_left + wall_width//2, wall_top - 20),
        ("Door Opening", door_left + door_width//2, door_top - 20),
        ("Height: 400", wall_left - 50, wall_top + wall_height//2),
        ("Width: 600", wall_left + wall_width//2, wall_top + wall_height + 20)
    ]
    
    for text, x, y in labels:
        if font:
            draw.text((x - 50, y), text, fill='black', font=font)
        else:
            draw.text((x - 50, y), text, fill='black')
    
    # Save the image
    filename = "test_door.jpg"
    image.save(filename, "JPEG", quality=95)
    print(f"✅ Test door image created: {filename}")
    print(f"📏 Image size: {width}x{height} pixels")
    print(f"📁 Saved as: {os.path.abspath(filename)}")
    
    return filename

def main():
    """Create test images for visual context analysis."""
    print("🎨 Creating Test Images for Visual Context Analysis")
    print("=" * 60)
    
    # Create wall image
    print("\n1. Creating wall with window opening...")
    wall_image = create_test_wall_image()
    
    # Create door image
    print("\n2. Creating wall with door opening...")
    door_image = create_test_door_image()
    
    print("\n✅ All test images created successfully!")
    print("\n📋 Usage Examples:")
    print(f"  python main.py  # Then use '{wall_image}' when prompted for image")
    print(f"  python -c \"from main import run_with_image; import asyncio; asyncio.run(run_with_image('create a window on the selected wall', '{wall_image}'))\"")
    print(f"  python -c \"from main import run_with_image; import asyncio; asyncio.run(run_with_image('create a door on the selected wall', '{door_image}'))\"")
    
    print("\n🔍 Test the visual context analysis:")
    print(f"  python test_visual_context.py")

if __name__ == "__main__":
    main() 