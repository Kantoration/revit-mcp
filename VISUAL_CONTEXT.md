# Visual Context Analysis

## Overview

The Visual Context Analysis feature uses OpenAI's CLIP (Contrastive Language-Image Pre-training) model to enhance the AI's understanding of Revit instructions by analyzing images provided by users. This allows the AI to generate more accurate and contextually relevant Revit code.

## Features

- **Image-Text Similarity**: Analyzes how well an image matches a text instruction
- **Visual Context Integration**: Enhances AI planning with visual information
- **Flexible Device Support**: Choose between CPU and GPU processing
- **Configurable Thresholds**: Adjust similarity requirements
- **Graceful Fallbacks**: Works even when CLIP is not available

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# Enable/disable visual context analysis
VISUAL_CONTEXT_ENABLED=true

# CLIP similarity threshold (0.0 to 1.0)
CLIP_SIMILARITY_THRESHOLD=0.3

# CLIP device choice (auto, cpu, gpu, cuda)
CLIP_DEVICE_CHOICE=auto
```

### Device Configuration Options

| Option | Description | Use Case |
|--------|-------------|----------|
| `auto` | Automatically choose best available device | Best for most users |
| `cpu` | Force CPU usage | Servers without GPU, guaranteed compatibility |
| `gpu` | Force GPU usage (if available) | Maximum performance when GPU available |
| `cuda` | Force CUDA usage (if available) | Same as `gpu` |

### Similarity Thresholds

| Threshold | Description | Behavior |
|-----------|-------------|----------|
| 0.3 (default) | Low threshold | Accepts most relevant images |
| 0.5 | Medium threshold | Requires stronger visual-text alignment |
| 0.7 | High threshold | Requires very strong visual-text alignment |

## Installation

### Basic Installation (CPU only)
```bash
pip install torch transformers Pillow
```

### GPU Installation (for CUDA acceleration)
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install transformers Pillow
```

### Verify Installation
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage

### Basic Usage

```python
from visual_context import analyze_visual_context

# Analyze an image with an instruction
context, similarity = analyze_visual_context(
    image_path="wall_image.jpg",
    user_instruction="create a window on the selected wall"
)

print(f"Similarity: {similarity:.3f}")
print(f"Context: {context}")
```

### Advanced Usage with Device Choice

```python
from visual_context import VisualContextAnalyzer

# Create analyzer with specific device
analyzer = VisualContextAnalyzer(device_choice="gpu")

# Get device information
device_info = analyzer.get_device_info()
print(f"Using device: {device_info['actual_device']}")

# Analyze image
context, similarity = analyzer.analyze_visual_context(
    image_path="wall_image.jpg",
    user_instruction="create a window on the selected wall"
)
```

### Integration with Main Application

The visual context analysis is automatically integrated into the main Revit AI Assistant. When you provide an image along with your instruction, the system will:

1. Analyze the image-text similarity
2. Generate visual context description
3. Provide visual suggestions
4. Enhance the AI planning process

## Performance Considerations

### GPU vs CPU Performance

- **GPU (CUDA)**: 5-10x faster than CPU for visual analysis
- **CPU**: Slower but works on any system
- **Memory**: CLIP model requires ~1GB RAM
- **First run**: Model download may take a few minutes

### Device Selection Logic

1. **auto**: Uses GPU if CUDA is available, falls back to CPU
2. **cpu**: Always uses CPU (slower but guaranteed to work)
3. **gpu/cuda**: Uses GPU if available, falls back to CPU with warning

## Example Configurations

### Development Mode (CPU)
```bash
VISUAL_CONTEXT_ENABLED=true
CLIP_DEVICE_CHOICE=cpu
CLIP_SIMILARITY_THRESHOLD=0.3
```

### Production Mode (GPU)
```bash
VISUAL_CONTEXT_ENABLED=true
CLIP_DEVICE_CHOICE=gpu
CLIP_SIMILARITY_THRESHOLD=0.5
```

### High Precision Mode
```bash
VISUAL_CONTEXT_ENABLED=true
CLIP_DEVICE_CHOICE=auto
CLIP_SIMILARITY_THRESHOLD=0.7
```

### Disabled Mode
```bash
VISUAL_CONTEXT_ENABLED=false
```

## Testing

Run the device configuration test:

```bash
python test_clip_devices.py
```

This will test all device configurations and show detailed information about your setup.

## Troubleshooting

### Common Issues

1. **CLIP not available**: Install dependencies with `pip install torch transformers Pillow`
2. **CUDA not available**: Install PyTorch with CUDA support or use CPU mode
3. **Out of memory**: Use CPU mode or reduce batch size
4. **Slow performance**: Use GPU mode if available

### Error Messages

- `CLIP dependencies not available`: Install required packages
- `GPU requested but CUDA not available`: Use CPU mode or install CUDA
- `Visual analysis failed`: Check image path and format

## API Reference

### VisualContextAnalyzer

```python
class VisualContextAnalyzer:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device_choice="auto")
    
    def analyze_visual_context(self, image_path: str, user_instruction: str) -> Tuple[str, float]
    def get_device_info(self) -> Dict[str, Any]
    def get_visual_suggestions(self, image_path: str, instruction: str) -> list
```

### Convenience Functions

```python
def analyze_visual_context(image_path: str, user_instruction: str, device_choice: str = "auto") -> Tuple[str, float]
def get_global_analyzer(device_choice: str = "auto") -> VisualContextAnalyzer
def reset_global_analyzer(device_choice: str = "auto") -> VisualContextAnalyzer
```

## Best Practices

1. **Use appropriate device**: GPU for performance, CPU for compatibility
2. **Set reasonable thresholds**: 0.3 for development, 0.5-0.7 for production
3. **Provide relevant images**: Images should match the instruction context
4. **Monitor performance**: Use device info to verify configuration
5. **Handle fallbacks**: Always check if CLIP is available before using

## Future Enhancements

- Support for multiple image analysis
- Custom CLIP model fine-tuning
- Integration with other vision models
- Real-time image capture from Revit
- Batch processing for multiple instructions 