#!/usr/bin/env python3
"""
Test script for visual context analysis using CLIP.
This script demonstrates how the visual context analysis works with the Revit AI Assistant.
"""

import os
import asyncio
from visual_context import VisualContextAnalyzer, analyze_visual_context

def test_visual_analysis():
    """Test the visual context analysis functionality."""
    print("🔍 Testing Visual Context Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    print("1. Initializing CLIP analyzer...")
    analyzer = VisualContextAnalyzer()
    
    if not analyzer.model:
        print("❌ CLIP model failed to initialize")
        print("Make sure you have installed the required dependencies:")
        print("pip install transformers torch Pillow")
        return
    
    print("✅ CLIP analyzer initialized successfully")
    
    # Test with sample instruction
    test_instruction = "create a window on the selected wall"
    
    print(f"\n2. Testing with instruction: '{test_instruction}'")
    
    # Test without image (should handle gracefully)
    print("\n📸 Testing without image...")
    context, score = analyzer.analyze_visual_context("nonexistent.jpg", test_instruction)
    print(f"Context: {context}")
    print(f"Score: {score:.3f}")
    
    # Test visual suggestions
    print("\n💡 Testing visual suggestions...")
    suggestions = analyzer.get_visual_suggestions("nonexistent.jpg", test_instruction)
    print("Suggestions:")
    for suggestion in suggestions:
        print(f"  - {suggestion}")
    
    print("\n✅ Visual context analysis test completed!")

def test_multiple_images():
    """Test multiple image analysis."""
    print("\n🖼️  Testing Multiple Image Analysis")
    print("=" * 50)
    
    analyzer = VisualContextAnalyzer()
    
    if not analyzer.model:
        print("❌ CLIP model not available")
        return
    
    # Test with multiple (nonexistent) images
    test_images = ["image1.jpg", "image2.jpg", "image3.jpg"]
    test_instruction = "create a door on the wall"
    
    print(f"Testing with instruction: '{test_instruction}'")
    print(f"Images: {test_images}")
    
    result = analyzer.analyze_multiple_images(test_images, test_instruction)
    
    print(f"\nResults:")
    print(f"  Visual Context: {result['visual_context']}")
    print(f"  Average Similarity: {result['similarity_score']:.3f}")
    print(f"  Image Count: {result['image_count']}")
    
    if result['best_match']:
        print(f"  Best Match: {result['best_match']['image_path']}")
        print(f"  Best Score: {result['best_match']['similarity_score']:.3f}")

async def test_integration():
    """Test integration with the main application."""
    print("\n🔗 Testing Integration with Main Application")
    print("=" * 50)
    
    try:
        from main import run_with_image
        
        # Test with a sample instruction and image
        instruction = "create a window on the selected wall"
        image_path = "sample_wall.jpg"  # This would be a real image in practice
        
        print(f"Testing with instruction: '{instruction}'")
        print(f"Image path: {image_path}")
        
        # Note: This will fail gracefully if the image doesn't exist
        await run_with_image(instruction, image_path)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the correct directory")
    except Exception as e:
        print(f"❌ Integration test failed: {e}")

def create_sample_usage():
    """Create sample usage examples."""
    print("\n📝 Sample Usage Examples")
    print("=" * 50)
    
    examples = [
        {
            "instruction": "create a window on the selected wall",
            "image_description": "Image showing a wall with space for a window",
            "expected_context": "High similarity - visual context confirms wall presence"
        },
        {
            "instruction": "create a door on the wall",
            "image_description": "Image showing a wall with door opening",
            "expected_context": "High similarity - visual context shows door location"
        },
        {
            "instruction": "create a floor at level 1",
            "image_description": "Image showing building foundation",
            "expected_context": "Medium similarity - visual context shows floor elements"
        },
        {
            "instruction": "get information about selected elements",
            "image_description": "Image showing various Revit elements",
            "expected_context": "Medium similarity - visual context shows selectable elements"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. Instruction: '{example['instruction']}'")
        print(f"   Image: {example['image_description']}")
        print(f"   Expected: {example['expected_context']}")

def main():
    """Main test function."""
    print("🚀 Visual Context Analysis Test Suite")
    print("=" * 60)
    
    # Test basic functionality
    test_visual_analysis()
    
    # Test multiple images
    test_multiple_images()
    
    # Show usage examples
    create_sample_usage()
    
    # Test integration (async)
    print("\n" + "=" * 60)
    print("🔗 Running integration test...")
    asyncio.run(test_integration())
    
    print("\n✅ All tests completed!")
    print("\n📋 Next Steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Add images to test with: python test_visual_context.py")
    print("3. Run main application: python main.py")
    print("4. Try with real images for best results")

if __name__ == "__main__":
    main() 