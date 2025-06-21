#!/usr/bin/env python3
"""
Test script to demonstrate CLIP device configuration options.
This script shows how to use different device choices for visual context analysis.
"""

import os
import asyncio
from visual_context import VisualContextAnalyzer, get_global_analyzer, reset_global_analyzer

def test_device_configuration():
    """Test different device configurations."""
    print("🔍 CLIP Device Configuration Test")
    print("=" * 50)
    
    # Test image path (you can change this to a real image)
    test_image = "test_wall.jpg"  # Change this to your image path
    test_instruction = "create a window on the selected wall"
    
    # Check if test image exists
    if not os.path.exists(test_image):
        print(f"⚠️  Test image '{test_image}' not found. Skipping actual analysis.")
        print("   Create a test image or change the path to test full functionality.")
        test_image = None
    
    # Test different device configurations
    device_configs = [
        ("auto", "Automatic device selection"),
        ("cpu", "Force CPU usage"),
        ("gpu", "Force GPU usage (if available)"),
        ("cuda", "Force CUDA usage (if available)")
    ]
    
    for device_choice, description in device_configs:
        print(f"\n📱 Testing: {description}")
        print(f"   Device choice: {device_choice}")
        
        try:
            # Create analyzer with specific device choice
            analyzer = VisualContextAnalyzer(device_choice=device_choice)
            
            # Get device information
            device_info = analyzer.get_device_info()
            
            print(f"   ✅ Device initialized successfully")
            print(f"   📊 Device info:")
            print(f"      - Choice: {device_info['device_choice']}")
            print(f"      - Actual: {device_info['actual_device']}")
            print(f"      - CLIP available: {device_info['clip_available']}")
            print(f"      - Model initialized: {device_info['model_initialized']}")
            
            if device_info['clip_available']:
                print(f"      - PyTorch version: {device_info['torch_version']}")
                print(f"      - CUDA available: {device_info['cuda_available']}")
                
                if device_info['cuda_available']:
                    print(f"      - GPU: {device_info['gpu_name']}")
                    print(f"      - GPU Memory: {device_info['gpu_memory_gb']:.1f} GB")
            
            # Test actual analysis if image exists
            if test_image and device_info['model_initialized']:
                print(f"   🔍 Testing analysis with image: {test_image}")
                context, similarity = analyzer.analyze_visual_context(test_image, test_instruction)
                print(f"   📈 Similarity score: {similarity:.3f}")
                print(f"   📝 Context: {context[:100]}...")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
        
        print("-" * 40)

def test_global_analyzer():
    """Test global analyzer with different device choices."""
    print("\n🌐 Global Analyzer Test")
    print("=" * 50)
    
    # Test resetting global analyzer with different devices
    devices = ["auto", "cpu", "gpu"]
    
    for device in devices:
        print(f"\n🔄 Testing global analyzer with device: {device}")
        try:
            # Reset global analyzer with new device choice
            analyzer = reset_global_analyzer(device_choice=device)
            device_info = analyzer.get_device_info()
            
            print(f"   ✅ Global analyzer reset successfully")
            print(f"   📊 Device: {device_info['actual_device']} (choice: {device_info['device_choice']})")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

def show_installation_guide():
    """Show installation guide for CLIP dependencies."""
    print("\n📦 CLIP Installation Guide")
    print("=" * 50)
    print("To enable CLIP visual context analysis, install the required dependencies:")
    print()
    print("1. Basic installation (CPU only):")
    print("   pip install torch transformers Pillow")
    print()
    print("2. CUDA installation (for GPU acceleration):")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("   pip install transformers Pillow")
    print()
    print("3. Check CUDA availability:")
    print("   python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\"")
    print()
    print("4. Environment variables for configuration:")
    print("   VISUAL_CONTEXT_ENABLED=true")
    print("   CLIP_DEVICE_CHOICE=auto  # or cpu, gpu, cuda")
    print("   CLIP_SIMILARITY_THRESHOLD=0.3")

def main():
    """Main test function."""
    print("🚀 CLIP Device Configuration Test Suite")
    print("=" * 60)
    
    # Test device configurations
    test_device_configuration()
    
    # Test global analyzer
    test_global_analyzer()
    
    # Show installation guide
    show_installation_guide()
    
    print("\n✅ Test completed!")
    print("\n💡 Tips:")
    print("- Use 'auto' for best automatic device selection")
    print("- Use 'cpu' if you don't have a GPU or want guaranteed compatibility")
    print("- Use 'gpu' or 'cuda' for maximum performance (if GPU available)")
    print("- Set CLIP_DEVICE_CHOICE in your .env file for persistent configuration")

if __name__ == "__main__":
    main() 