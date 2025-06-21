# Configuration Guide

## Environment Variables

Add these to your `.env` file:

### Required
```bash
# Groq API Key
GROQ_API_KEY=your_groq_api_key_here

# LangSmith Configuration (optional but recommended)
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=revit-ai-assistant
```

### Fallback Configuration
```bash
# Fallback behavior when JSON parsing fails
FALLBACK_MODE=auto          # auto, manual, abort
SAVE_FAILED_RESPONSES=true  # true, false
```

### MCP Execution Configuration
```bash
# Enable/disable MCP execution
MCP_ENABLED=false           # true, false

# Enable async execution (requires MCP_ENABLED=true)
ASYNC_EXECUTION=false       # true, false
```

### Visual Context Configuration
```bash
# Enable/disable CLIP-based visual context analysis
VISUAL_CONTEXT_ENABLED=true # true, false

# CLIP similarity threshold (0.0 to 1.0, higher = more strict matching)
CLIP_SIMILARITY_THRESHOLD=0.3  # 0.3, 0.5, 0.7, etc.
```

## Configuration Options

### FALLBACK_MODE
- **auto** (default): Creates fallback method and continues
- **manual**: Saves responses for manual review and stops
- **abort**: Saves responses and exits cleanly

### MCP_ENABLED
- **false** (default): Demo mode - no actual Revit execution
- **true**: Enables actual MCP server execution

### ASYNC_EXECUTION
- **false** (default): Sequential execution
- **true**: Concurrent execution (max 3 tasks at once)

### VISUAL_CONTEXT_ENABLED
- **true** (default): Enables CLIP-based visual context analysis
- **false**: Disables visual context analysis (text-only mode)

### CLIP_SIMILARITY_THRESHOLD
- **0.3** (default): Minimum similarity score for visual context to be considered relevant
- **0.5**: Medium threshold - requires stronger visual-text alignment
- **0.7**: High threshold - requires very strong visual-text alignment

## Example Configurations

### Demo Mode (Default)
```bash
MCP_ENABLED=false
ASYNC_EXECUTION=false
FALLBACK_MODE=auto
VISUAL_CONTEXT_ENABLED=true
CLIP_SIMILARITY_THRESHOLD=0.3
```

### Production Mode
```bash
MCP_ENABLED=true
ASYNC_EXECUTION=true
FALLBACK_MODE=auto
VISUAL_CONTEXT_ENABLED=true
CLIP_SIMILARITY_THRESHOLD=0.5
```

### Debug Mode
```bash
MCP_ENABLED=false
ASYNC_EXECUTION=false
FALLBACK_MODE=manual
SAVE_FAILED_RESPONSES=true
VISUAL_CONTEXT_ENABLED=true
CLIP_SIMILARITY_THRESHOLD=0.3
```

### Text-Only Mode
```bash
MCP_ENABLED=false
ASYNC_EXECUTION=false
FALLBACK_MODE=auto
VISUAL_CONTEXT_ENABLED=false
``` 