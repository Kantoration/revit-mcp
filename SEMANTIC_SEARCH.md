# Semantic Search for Method Reuse

## Overview

The Revit AI Assistant now includes advanced semantic search capabilities that significantly enhance method reuse. Instead of simple keyword matching, the system uses embeddings and similarity matching to find the most relevant existing methods for any given instruction.

## Features

### 🔍 **Semantic Understanding**
- Uses sentence transformers to understand the meaning behind user instructions
- Matches intent rather than just keywords
- Handles synonyms and different phrasings

### 📊 **Similarity Scoring**
- Calculates cosine similarity between query and method embeddings
- Configurable similarity threshold (default: 0.7)
- Returns ranked results with similarity scores

### 🛡️ **Fallback System**
- Automatic fallback to keyword search if embeddings fail
- Graceful degradation ensures the system always works
- No single point of failure

### 🔄 **Automatic Updates**
- Embeddings are automatically updated when new methods are added
- Real-time search index maintenance
- No manual intervention required

## How It Works

### 1. **Embedding Generation**
```python
# Each method is converted to searchable text
search_text = method_name + description + keywords + parameters + code_preview
embedding = embeddings_model.encode(search_text)
```

### 2. **Query Processing**
```python
# User instruction is converted to embedding
query_embedding = embeddings_model.encode(user_instruction)
```

### 3. **Similarity Calculation**
```python
# Cosine similarity between query and all method embeddings
similarity = np.dot(query_embedding, method_embedding)
```

### 4. **Ranking and Filtering**
```python
# Results are ranked by similarity and filtered by threshold
filtered_results = [method for score, method in similarities if score >= threshold]
```

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# Similarity threshold (0.0 to 1.0, higher = more strict matching)
SIMILARITY_THRESHOLD=0.7
```

### Dependencies

The semantic search requires these additional packages:

```bash
pip install sentence-transformers numpy torch
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage Examples

### Basic Search
```python
from main import search_method_library

# Search for methods related to creating windows
results = search_method_library("create a window on the selected wall")
```

### Testing the System
```bash
# Run the test script to see semantic search in action
python test_semantic_search.py
```

## Example Queries and Matches

| Query | Expected Match | Reason |
|-------|---------------|---------|
| "create a window on the selected wall" | `create_window_on_wall` | Direct semantic match |
| "add a door to the wall" | `create_door_on_wall` | Synonym understanding |
| "get information about selected elements" | `get_element_information` | Intent matching |
| "insert an opening in the wall" | `create_window_on_wall` or `create_door_on_wall` | Concept matching |
| "retrieve element properties" | `get_element_information` | Synonym matching |

## Integration with AI Workflow

### 1. **Pre-Search**
Before generating new code, the system searches for existing methods:
```python
matching_methods = search_method_library(user_instruction)
```

### 2. **Enhanced Prompting**
Search results are included in the AI prompt:
```
**Semantically Similar Methods Found:**
1. **create_window_on_wall**: Creates a window on a selected wall
   Parameters: wall_position, window_type, height
   Keywords: window, wall, create, opening, insert
```

### 3. **Method Reuse Decision**
The AI can choose to:
- Use an existing method with parameters
- Create a new method if no suitable match exists
- Combine multiple existing methods

## Performance Benefits

### 🚀 **Faster Development**
- Reuse existing, tested methods
- Reduce code duplication
- Accelerate feature development

### 🎯 **Better Accuracy**
- Semantic understanding reduces false negatives
- Context-aware matching
- Improved method discovery

### 🔧 **Maintainability**
- Centralized method library
- Consistent code patterns
- Easier debugging and updates

## Troubleshooting

### Embedding Initialization Fails
```
Failed to initialize embeddings: [error]
Falling back to keyword-based search
```
**Solution**: Check internet connection and try again. The system will work with keyword search as fallback.

### No Methods Found
```
No semantically similar methods found above threshold
```
**Solutions**:
1. Lower the `SIMILARITY_THRESHOLD` in your `.env` file
2. Add more methods to the library
3. Use more descriptive method names and keywords

### Slow Performance
**Solutions**:
1. Use GPU if available (modify `model_kwargs={'device': 'cuda'}`)
2. Reduce the number of methods in the library
3. Use more specific queries

## Advanced Configuration

### Custom Embedding Model
```python
# In MethodSearchEngine._initialize_embeddings()
self.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",  # Larger, more accurate
    model_kwargs={'device': 'cuda'},  # Use GPU
    encode_kwargs={'normalize_embeddings': True}
)
```

### Custom Search Text Creation
```python
def _create_search_text(self, method):
    # Customize how method information is combined
    search_parts = [
        method.get('method_name', ''),
        method.get('description', ''),
        ' '.join(method.get('keywords', [])),
        method.get('csharp_code', '')[:1000]  # More code context
    ]
    return ' '.join(filter(None, search_parts))
```

## Future Enhancements

- **Hybrid Search**: Combine semantic and keyword search
- **Learning**: Improve embeddings based on usage patterns
- **Clustering**: Group similar methods automatically
- **Versioning**: Track method evolution over time
- **Collaborative Filtering**: Suggest methods based on user patterns 