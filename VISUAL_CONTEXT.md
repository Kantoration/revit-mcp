# Visual Context Analysis with CLIP

## Overview

The Revit AI Assistant now includes **CLIP-based visual context analysis** that significantly enhances the AI's understanding by combining visual and textual information. This feature allows users to provide images alongside their instructions, enabling the AI to make more informed decisions about Revit operations.

## 🎯 Key Benefits

### **Enhanced Understanding**
- **Visual-text alignment**: CLIP analyzes how well the image matches the instruction
- **Context-aware planning**: AI considers visual elements when generating code
- **Reduced ambiguity**: Visual context helps clarify user intent

### **Improved Accuracy**
- **Better method selection**: Visual context helps choose the right existing methods
- **More precise parameters**: Image analysis can suggest specific dimensions or types
- **Reduced errors**: Visual verification reduces misinterpretation of instructions

### **Flexible Integration**
- **Optional feature**: Can be enabled/disabled via configuration
- **Graceful fallback**: Works seamlessly even without images
- **Multiple image support**: Can analyze multiple images for comprehensive context

## 🔧 How It Works

### 1. **Image Analysis Pipeline**
```
User Input → CLIP Model → Similarity Score → Context Generation → Enhanced Prompt
```

### 2. **CLIP Processing**
- **Image encoding**: Converts image to high-dimensional vector
- **Text encoding**: Converts instruction to matching vector space
- **Similarity calculation**: Computes cosine similarity between vectors
- **Context generation**: Creates descriptive context based on similarity

### 3. **Integration with LangChain**
- **Enhanced prompts**: Visual context is added to the AI prompt
- **Guided planning**: AI receives specific guidelines based on similarity scores
- **Method selection**: Visual context influences existing method reuse decisions

## 📊 Similarity Score Interpretation

| Score Range | Meaning | Planning Strategy |
|-------------|---------|-------------------|
| **0.7 - 1.0** | Strong match | Use visual context confidently |
| **0.5 - 0.7** | Moderate match | Consider visual elements |
| **0.3 - 0.5** | Weak match | Use as hints, verify assumptions |
| **0.0 - 0.3** | Poor match | Rely primarily on text |

## 🚀 Usage Examples

### Basic Usage
```python
from visual_context import analyze_visual_context

# Analyze image with instruction
context, score = analyze_visual_context("wall_image.jpg", "create a window on the selected wall")
print(f"Context: {context}")
print(f"Similarity: {score:.3f}")
```

### Integration with Main Application
```python
from main import run_with_image

# Run with image and instruction
await run_with_image("create a door on the wall", "door_location.jpg")
```

### Multiple Image Analysis
```python
from visual_context import VisualContextAnalyzer

analyzer = VisualContextAnalyzer()
images = ["view1.jpg", "view2.jpg", "detail.jpg"]
result = analyzer.analyze_multiple_images(images, "create a window on the selected wall")
```

## ⚙️ Configuration

### Environment Variables
```bash
# Enable/disable visual context analysis
VISUAL_CONTEXT_ENABLED=true

# CLIP similarity threshold
CLIP_SIMILARITY_THRESHOLD=0.3
```

### Configuration Options

#### VISUAL_CONTEXT_ENABLED
- **true** (default): Enables CLIP-based visual analysis
- **false**: Disables visual context (text-only mode)

#### CLIP_SIMILARITY_THRESHOLD
- **0.3** (default): Minimum similarity for relevance
- **0.5**: Medium threshold - stronger alignment required
- **0.7**: High threshold - very strong alignment required

## 📋 Supported Image Formats

- **JPEG** (.jpg, .jpeg)
- **PNG** (.png)
- **BMP** (.bmp)
- **TIFF** (.tiff)
- **WebP** (.webp)

## 🔍 Visual Context Types

### **Wall Operations**
- Wall type identification
- Height and thickness estimation
- Material properties
- Opening locations

### **Opening Operations**
- Window/door sizing
- Positioning relative to wall
- Type and style identification
- Frame details

### **Floor Operations**
- Floor type identification
- Boundary geometry
- Level information
- Material properties

### **Element Information**
- Element type identification
- Property extraction
- Geometry analysis
- Selection guidance

## 🛠️ Technical Implementation

### **CLIP Model**
- **Model**: `openai/clip-vit-base-patch32`
- **Device**: Automatic CUDA/CPU detection
- **Memory**: ~150MB model size
- **Speed**: ~100ms per image (GPU), ~500ms (CPU)

### **Embedding Process**
```python
# Image preprocessing
image = Image.open(image_path).convert("RGB")
inputs = processor(text=[instruction], images=image, return_tensors="pt")

# Model inference
outputs = model(**inputs)
image_embeds = outputs.image_embeds
text_embeds = outputs.text_embeds

# Similarity calculation
similarity = torch.cosine_similarity(image_embeds, text_embeds)[0].item()
```

### **Context Generation**
```python
def generate_context(similarity, instruction):
    if similarity > 0.7:
        return f"The image strongly matches '{instruction}'. Proceed with confidence."
    elif similarity > 0.5:
        return f"The image moderately matches '{instruction}'. Consider visual elements."
    # ... more conditions
```

## 📈 Performance Considerations

### **Memory Usage**
- **Model loading**: ~150MB RAM
- **Per-image processing**: ~50MB temporary memory
- **Batch processing**: Scales linearly with image count

### **Processing Speed**
- **GPU acceleration**: 10-50x faster than CPU
- **Batch processing**: Efficient for multiple images
- **Caching**: Global analyzer instance for reuse

### **Accuracy Trade-offs**
- **Higher thresholds**: More precise but fewer matches
- **Lower thresholds**: More matches but potential false positives
- **Model size**: Larger models = better accuracy but slower inference

## 🧪 Testing and Validation

### **Test Script**
```bash
# Run visual context tests
python test_visual_context.py
```

### **Test Coverage**
- Basic functionality testing
- Multiple image analysis
- Integration with main application
- Error handling and fallbacks
- Performance benchmarking

### **Validation Examples**
| Instruction | Image Content | Expected Score | Expected Context |
|-------------|---------------|----------------|------------------|
| "create window" | Wall with window opening | 0.7-0.9 | Strong match, proceed confidently |
| "create door" | Wall with door frame | 0.6-0.8 | Moderate match, consider visual elements |
| "create floor" | Building foundation | 0.4-0.6 | Weak match, use as hints |
| "get element info" | Various Revit elements | 0.3-0.5 | Some relevance, verify assumptions |

## 🔮 Future Enhancements

### **Planned Features**
- **Object detection**: Identify specific Revit elements in images
- **Measurement extraction**: Estimate dimensions from images
- **Style recognition**: Identify architectural styles and preferences
- **Multi-modal prompts**: Combine image and text in prompts

### **Advanced Capabilities**
- **Video analysis**: Process video clips for dynamic context
- **3D understanding**: Analyze 3D renders and models
- **Real-time analysis**: Live camera feed integration
- **Custom training**: Fine-tune CLIP for Revit-specific tasks

## 🐛 Troubleshooting

### **Common Issues**

#### CLIP Model Not Loading
```bash
# Install dependencies
pip install transformers torch Pillow

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

#### Low Similarity Scores
- **Check image quality**: Ensure clear, relevant images
- **Adjust threshold**: Lower CLIP_SIMILARITY_THRESHOLD
- **Verify instruction**: Ensure clear, specific instructions

#### Memory Issues
- **Use CPU**: Set device to CPU if GPU memory is limited
- **Reduce batch size**: Process fewer images simultaneously
- **Clear cache**: Restart application to free memory

### **Debug Mode**
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py
```

## 📚 Additional Resources

### **CLIP Documentation**
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/model_doc/clip)
- [OpenAI CLIP](https://github.com/openai/CLIP)

### **Related Technologies**
- **LangChain**: Chain orchestration and prompting
- **Revit API**: C# code generation and execution
- **Semantic Search**: Method library search and reuse

### **Best Practices**
- **Image quality**: Use clear, well-lit images
- **Relevance**: Ensure images match the instruction
- **Multiple views**: Provide different angles for complex operations
- **Consistent naming**: Use descriptive image filenames 