#!/usr/bin/env python3
"""
Google Colab Setup Script for Revit AI Assistant with CUDA
This script sets up the environment for running the visual context analysis on Colab GPUs.
"""

import os
import subprocess
import sys

def install_dependencies():
    """Install required packages for CUDA support."""
    print("🔧 Installing dependencies for CUDA support...")
    
    packages = [
        "torch",
        "torchvision", 
        "transformers",
        "Pillow",
        "langchain",
        "langchain-core",
        "langchain-groq",
        "langchain-community",
        "langchain-huggingface",
        "python-dotenv",
        "requests",
        "sentence-transformers",
        "numpy"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_cuda():
    """Check CUDA availability and print GPU info."""
    print("\n🔍 Checking CUDA availability...")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("❌ CUDA not available - using CPU")
            
    except ImportError:
        print("❌ PyTorch not installed")

def setup_environment():
    """Set up environment variables for Colab."""
    print("\n⚙️ Setting up environment...")
    
    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid warnings
    
    print("✅ Environment configured")

def download_models():
    """Pre-download models for faster startup."""
    print("\n📥 Pre-downloading models...")
    
    try:
        from transformers import CLIPProcessor, CLIPModel
        from sentence_transformers import SentenceTransformer
        
        # Download CLIP model
        print("Downloading CLIP model...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Download sentence transformer
        print("Downloading sentence transformer...")
        sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        print("✅ Models downloaded successfully")
        
    except Exception as e:
        print(f"❌ Error downloading models: {e}")

def create_colab_notebook():
    """Create a Colab notebook template."""
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"id": "setup"},
                "source": [
                    "# Revit AI Assistant with Visual Context (CUDA)\n\nThis notebook runs the Revit AI Assistant with CLIP visual context analysis on Colab GPU."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "install"},
                "outputs": [],
                "source": [
                    "# Install dependencies\n",
                    "!pip install torch torchvision transformers Pillow langchain langchain-core langchain-groq langchain-community langchain-huggingface python-dotenv requests sentence-transformers numpy"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "check_gpu"},
                "outputs": [],
                "source": [
                    "# Check GPU availability\n",
                    "import torch\n",
                    "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
                    "if torch.cuda.is_available():\n",
                    "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n",
                    "    print(f\"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "upload_files"},
                "outputs": [],
                "source": [
                    "# Upload your files\n",
                    "from google.colab import files\n",
                    "uploaded = files.upload()\n",
                    "\n",
                    "# Extract if needed\n",
                    "!unzip -q langchain-starter.zip 2>/dev/null || echo \"No zip file found\""
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "run_assistant"},
                "outputs": [],
                "source": [
                    "# Run the assistant\n",
                    "import os\n",
                    "os.chdir('langchain-starter')\n",
                    "\n",
                    "# Set your API keys\n",
                    "os.environ['GROQ_API_KEY'] = 'your_groq_api_key_here'\n",
                    "os.environ['LANGCHAIN_API_KEY'] = 'your_langsmith_api_key_here'\n",
                    "\n",
                    "# Run with visual context\n",
                    "from main import run_with_image\n",
                    "import asyncio\n",
                    "\n",
                    "# Test with a sample image\n",
                    "await run_with_image(\"create a window on the selected wall\", \"test_wall.jpg\")"
                ]
            }
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    import json
    with open('revit_ai_assistant_colab.ipynb', 'w') as f:
        json.dump(notebook_content, f, indent=2)
    
    print("✅ Colab notebook created: revit_ai_assistant_colab.ipynb")

def main():
    """Main setup function."""
    print("🚀 Setting up Revit AI Assistant for CUDA/Colab")
    print("=" * 50)
    
    # Install dependencies
    install_dependencies()
    
    # Check CUDA
    check_cuda()
    
    # Setup environment
    setup_environment()
    
    # Download models
    download_models()
    
    # Create Colab notebook
    create_colab_notebook()
    
    print("\n✅ Setup complete!")
    print("\n📋 Next steps:")
    print("1. Upload 'revit_ai_assistant_colab.ipynb' to Google Colab")
    print("2. Enable GPU: Runtime → Change runtime type → GPU")
    print("3. Add your API keys to the notebook")
    print("4. Run the cells to test with CUDA acceleration")

if __name__ == "__main__":
    main() 