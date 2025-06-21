#!/usr/bin/env python3
"""
LangChain application with LangSmith integration for web-based visualization.
This version will send all traces to LangSmith for monitoring and debugging.
"""

import os
import json
import time
import asyncio
import numpy as np
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from mcp_client import execute_revit_code
from visual_context import VisualContextAnalyzer, get_global_analyzer

# Load environment variables from .env
load_dotenv()

# --- Configuration ---
FALLBACK_MODE = os.getenv("FALLBACK_MODE", "auto").lower()  # auto, manual, abort
SAVE_FAILED_RESPONSES = os.getenv("SAVE_FAILED_RESPONSES", "true").lower() == "true"
MCP_ENABLED = os.getenv("MCP_ENABLED", "false").lower() == "true"
ASYNC_EXECUTION = os.getenv("ASYNC_EXECUTION", "false").lower() == "true"
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))  # Minimum similarity score

# --- Visual Context Configuration ---
VISUAL_CONTEXT_ENABLED = os.getenv("VISUAL_CONTEXT_ENABLED", "true").lower() == "true"
CLIP_SIMILARITY_THRESHOLD = float(os.getenv("CLIP_SIMILARITY_THRESHOLD", "0.3"))  # CLIP similarity threshold
CLIP_DEVICE_CHOICE = os.getenv("CLIP_DEVICE_CHOICE", "auto").lower()  # CLIP device choice

# --- Centralized Logging and Timing ---
class Logger:
    """Centralized logging with timing capabilities."""
    
    def __init__(self):
        self.start_time = None
    
    def start_session(self):
        """Start a new session with timestamp."""
        self.start_time = datetime.now()
        print(f"🕐 Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def end_session(self):
        """End session and show duration."""
        if self.start_time:
            duration = datetime.now() - self.start_time
            print(f"🕐 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏱️  Total duration: {duration.total_seconds():.2f}s")
    
    def header(self, title):
        """Print a formatted header."""
        print("\n" + "="*60)
        print(f"🚀 {title}")
        print("="*60)
    
    def step(self, step_num, description):
        """Print a formatted step."""
        print(f"\n📋 Step {step_num}: {description}")
        print("-" * 40)
    
    def info(self, message):
        """Print info message."""
        print(f"ℹ️  {message}")
    
    def success(self, message):
        """Print success message."""
        print(f"✅ {message}")
    
    def warning(self, message):
        """Print warning message."""
        print(f"⚠️  {message}")
    
    def error(self, message):
        """Print error message."""
        print(f"❌ {message}")

# Global logger instance
logger = Logger()

# --- Semantic Search System ---
class MethodSearchEngine:
    """Semantic search engine for method library using embeddings."""
    
    def __init__(self):
        self.embeddings = None
        self.method_embeddings = {}
        self.method_library = []
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize the embedding model."""
        try:
            logger.info("Initializing semantic search embeddings...")
            # Use a lightweight model for local embedding generation
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.success("Semantic search engine initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize embeddings: {e}")
            logger.info("Falling back to keyword-based search")
            self.embeddings = None
    
    def _create_search_text(self, method):
        """Create searchable text from method definition."""
        search_parts = [
            method.get('method_name', ''),
            method.get('description', ''),
            ' '.join(method.get('keywords', [])),
            ' '.join(method.get('parameters', [])),
            method.get('csharp_code', '')[:500]  # First 500 chars of code for context
        ]
        return ' '.join(filter(None, search_parts))
    
    def update_embeddings(self, method_library):
        """Update embeddings for the method library."""
        if not self.embeddings:
            logger.warning("Cannot update embeddings - embeddings not initialized")
            return
        
        self.method_library = method_library
        self.method_embeddings = {}
        
        if not method_library:
            logger.info("No methods to embed")
            return
        
        try:
            logger.info(f"Generating embeddings for {len(method_library)} methods...")
            # Create search texts for all methods
            search_texts = [self._create_search_text(method) for method in method_library]
            
            # Generate embeddings
            logger.info("Creating embeddings...")
            embeddings_list = self.embeddings.embed_documents(search_texts)
            logger.info(f"Generated {len(embeddings_list)} embeddings")
            
            # Store embeddings with method info
            for i, method in enumerate(method_library):
                self.method_embeddings[method['method_name']] = {
                    'embedding': embeddings_list[i],
                    'method': method
                }
            
            logger.success(f"Updated embeddings for {len(method_library)} methods")
            
        except Exception as e:
            logger.warning(f"Failed to update embeddings: {e}")
            self.method_embeddings = {}
    
    def search_semantic(self, query, top_k=3):
        """Search for methods using semantic similarity."""
        if not self.embeddings:
            logger.warning("Embeddings not available, using keyword search")
            return self._fallback_keyword_search(query, top_k)
        
        if not self.method_embeddings:
            logger.warning("No method embeddings available, using keyword search")
            return self._fallback_keyword_search(query, top_k)
        
        try:
            # Generate embedding for the query
            logger.info(f"Generating embedding for query: '{query}'")
            query_embedding = self.embeddings.embed_query(query)
            logger.info("Query embedding generated successfully")
            
            # Calculate similarities
            similarities = []
            for method_name, data in self.method_embeddings.items():
                similarity = np.dot(query_embedding, data['embedding'])
                similarities.append((similarity, data['method']))
            
            # Sort by similarity and filter by threshold
            similarities.sort(key=lambda x: x[0], reverse=True)
            filtered_results = [
                (score, method) for score, method in similarities 
                if score >= SIMILARITY_THRESHOLD
            ]
            
            # Return top results
            results = [method for score, method in filtered_results[:top_k]]
            
            if results:
                logger.info(f"Found {len(results)} semantically similar methods")
                for i, (score, method) in enumerate(filtered_results[:top_k]):
                    logger.info(f"  {i+1}. {method['method_name']} (score: {score:.3f})")
            else:
                logger.info(f"No semantically similar methods found above threshold {SIMILARITY_THRESHOLD}")
                logger.info("Top similarity scores:")
                for i, (score, method) in enumerate(similarities[:3]):
                    logger.info(f"  {i+1}. {method['method_name']} (score: {score:.3f})")
            
            return results
            
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            logger.info("Falling back to keyword search...")
            return self._fallback_keyword_search(query, top_k)
    
    def _fallback_keyword_search(self, query, top_k=3):
        """Fallback to keyword-based search."""
        query_lower = query.lower()
        query_words = set(query_lower.split())  # Split into individual words
        matching_methods = []
        
        for method in self.method_library:
            score = 0
            
            # Check method name
            method_name_lower = method.get('method_name', '').lower()
            method_name_words = set(method_name_lower.replace('_', ' ').split())
            name_matches = len(query_words.intersection(method_name_words))
            score += name_matches * 3  # Higher weight for method name matches
            
            # Check description
            description_lower = method.get('description', '').lower()
            description_words = set(description_lower.split())
            desc_matches = len(query_words.intersection(description_words))
            score += desc_matches * 2  # Medium weight for description matches
            
            # Check keywords
            keywords = method.get('keywords', [])
            keyword_matches = 0
            for keyword in keywords:
                if keyword.lower() in query_words:
                    keyword_matches += 1
            score += keyword_matches * 2  # Medium weight for keyword matches
            
            # Check parameters
            parameters = method.get('parameters', [])
            param_matches = 0
            for param in parameters:
                if param.lower() in query_words:
                    param_matches += 1
            score += param_matches  # Lower weight for parameter matches
            
            # Add method if it has any matches
            if score > 0:
                matching_methods.append((score, method))
        
        # Sort by score (highest first) and return top results
        matching_methods.sort(key=lambda x: x[0], reverse=True)
        results = [method for score, method in matching_methods[:top_k]]
        
        logger.info(f"Keyword search found {len(results)} methods")
        if results:
            for i, (score, method) in enumerate(matching_methods[:top_k]):
                logger.info(f"  {i+1}. {method['method_name']} (score: {score})")
        
        return results

# Global search engine instance
search_engine = MethodSearchEngine()

# --- LangSmith Configuration ---
def setup_langsmith():
    """Set up LangSmith environment variables."""
    langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
    langsmith_project = os.getenv("LANGCHAIN_PROJECT", "revit-ai-assistant")
    langsmith_tracing = os.getenv("LANGCHAIN_TRACING_V2", "true")
    
    if not langsmith_api_key:
        logger.warning("LangSmith not configured. To enable web visualization:")
        logger.info("1. Go to https://smith.langchain.com/")
        logger.info("2. Sign up for a free account")
        logger.info("3. Get your API key from settings")
        logger.info("4. Add to your .env file:")
        logger.info("   LANGCHAIN_API_KEY=your_api_key_here")
        logger.info("   LANGCHAIN_PROJECT=revit-ai-assistant")
        logger.info("   LANGCHAIN_TRACING_V2=true")
        return False
    
    logger.success("LangSmith configured!")
    logger.info(f"🌐 View traces at: https://smith.langchain.com/")
    logger.info(f"📊 Project: {langsmith_project}")
    return True

# --- Visualization Functions ---
def print_plan_visualization(task_breakdown):
    """Print a visual representation of the task breakdown."""
    logger.header("📊 TASK BREAKDOWN VISUALIZATION")
    
    for step in task_breakdown:
        logger.info(f"\n{step['step_number']:2d}. {step['description']}")
        logger.info(f"    💻 C# Code Preview:")
        
        # Show first few lines of the C# code
        code_lines = step['csharp_code'].split('\\n')
        for i, line in enumerate(code_lines[:3]):  # Show first 3 lines
            if line.strip():
                logger.info(f"       {line}")
        if len(code_lines) > 3:
            logger.info(f"       ... ({len(code_lines) - 3} more lines)")
        logger.info("")

# --- Tool Library Management ---
def load_tool_library():
    """Loads the custom tool library from a JSON file."""
    try:
        with open("custom_tool_library.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_new_tool(tool_definition):
    """Saves a newly created tool to the library, avoiding duplicates."""
    library = load_tool_library()
    if any(tool['tool_name'] == tool_definition['tool_name'] for tool in library):
        logger.warning(f"Tool '{tool_definition['tool_name']}' already exists. Skipping save.")
        return
    library.append(tool_definition)
    with open("custom_tool_library.json", "w") as f:
        json.dump(library, f, indent=2)
    logger.success(f"New tool '{tool_definition['tool_name']}' saved to library.")

# --- Method Library Management ---
def load_method_library():
    """Loads the method library from a JSON file."""
    try:
        with open("method_library.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_method_to_library(method_definition):
    """Saves a newly created method to the library, avoiding duplicates."""
    library = load_method_library()
    
    # Check for duplicates based on method name and description
    if any(method['method_name'] == method_definition['method_name'] for method in library):
        logger.warning(f"Method '{method_definition['method_name']}' already exists in library. Skipping save.")
        return
    
    library.append(method_definition)
    with open("method_library.json", "w") as f:
        json.dump(library, f, indent=2)
    logger.success(f"New method '{method_definition['method_name']}' saved to library.")
    
    # Update search engine embeddings
    search_engine.update_embeddings(library)

def search_method_library(instruction):
    """Search the method library for existing methods that can handle the instruction."""
    return search_engine.search_semantic(instruction, top_k=3)

def create_method_library_description():
    """Creates a description of all available methods for the prompt."""
    library = load_method_library()
    if not library:
        return "No existing methods available."
    
    descriptions = []
    for method in library:
        desc = f"- `{method['method_name']}`: {method['description']}"
        if method.get('parameters'):
            desc += f" (Parameters: {', '.join(method['parameters'])})"
        descriptions.append(desc)
    
    return "\n".join(descriptions)

def initialize_method_search():
    """Initialize the method search engine with current library."""
    library = load_method_library()
    search_engine.update_embeddings(library)
    logger.info(f"Method search engine initialized with {len(library)} methods")

# --- MCP Execution Functions ---
class RevitExecutor:
    """Unified adapter for sync/async Revit code execution."""
    
    def __init__(self, async_mode=False):
        self.async_mode = async_mode
        self.semaphore = asyncio.Semaphore(3) if async_mode else None
    
    async def execute_single_async(self, code, description):
        """Execute single Revit code asynchronously."""
        try:
            logger.info(f"🔄 Executing: {description}")
            result = await asyncio.to_thread(execute_revit_code, code)
            logger.success(f"✅ Completed: {description}")
            return {"success": True, "result": result, "description": description}
        except Exception as e:
            logger.error(f"❌ Failed: {description} - {str(e)}")
            return {"success": False, "error": str(e), "description": description}
    
    def execute_single_sync(self, code, description):
        """Execute single Revit code synchronously."""
        try:
            logger.info(f"🔄 Executing: {description}")
            result = execute_revit_code(code)
            logger.success(f"✅ Completed: {description}")
            return {"success": True, "result": result, "description": description}
        except Exception as e:
            logger.error(f"❌ Failed: {description} - {str(e)}")
            return {"success": False, "error": str(e), "description": description}
    
    async def execute_batch_async(self, task_breakdown):
        """Execute multiple tasks asynchronously with concurrency control."""
        logger.info(f"🚀 Starting async execution of {len(task_breakdown)} tasks")
        
        # Create execution tasks
        tasks = []
        for i, step in enumerate(task_breakdown, 1):
            if step.get("csharp_code"):
                task = self.execute_single_async(step["csharp_code"], f"Step {i}: {step['description']}")
                tasks.append(task)
        
        # Execute with concurrency control
        async def execute_with_semaphore(task):
            async with self.semaphore:
                return await task
        
        # Execute all tasks
        results = await asyncio.gather(*[execute_with_semaphore(task) for task in tasks], return_exceptions=True)
        
        # Log results
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        failed = len(results) - successful
        
        logger.info(f"📊 Execution Summary: {successful} successful, {failed} failed")
        return results
    
    def execute_batch_sync(self, task_breakdown):
        """Execute multiple tasks synchronously."""
        logger.info(f"🚀 Starting sync execution of {len(task_breakdown)} tasks")
        
        results = []
        for i, step in enumerate(task_breakdown, 1):
            if step.get("csharp_code"):
                result = self.execute_single_sync(step["csharp_code"], f"Step {i}: {step['description']}")
                results.append(result)
        
        # Log results
        successful = sum(1 for r in results if r.get("success"))
        failed = len(results) - successful
        
        logger.info(f"📊 Execution Summary: {successful} successful, {failed} failed")
        return results
    
    async def execute(self, task_breakdown):
        """Unified execution method that handles both sync and async modes."""
        if self.async_mode:
            return await self.execute_batch_async(task_breakdown)
        else:
            return self.execute_batch_sync(task_breakdown)

def create_executor(async_mode=False):
    """Factory function to create a RevitExecutor instance."""
    return RevitExecutor(async_mode=async_mode)

# --- Main Application ---
def get_user_input():
    """Get user instruction and optional image path from input."""
    logger.header("🎯 REVIT AI ASSISTANT - INSTRUCTION INPUT")
    logger.info("Enter your Revit instruction below, or press Enter for default test instruction.")
    logger.info("Examples:")
    logger.info("  - create a window on the selected wall")
    logger.info("  - create a door on the selected wall")
    logger.info("  - create a floor at level 1")
    logger.info("  - get information about selected elements")
    logger.info("  - create a wall between two points")
    logger.info("-" * 60)
    
    user_input = input("Your instruction: ").strip()
    
    if not user_input:
        default_instruction = "create a window on the selected wall"
        logger.info(f"Using default instruction: '{default_instruction}'")
        return default_instruction, None
    
    # Ask for image path if visual context is enabled
    image_path = None
    if VISUAL_CONTEXT_ENABLED:
        logger.info("\n📸 Visual Context Analysis:")
        logger.info("You can provide an image path to enhance the AI's understanding.")
        logger.info("Press Enter to skip image input, or provide a path to an image file.")
        
        image_input = input("Image path (optional): ").strip()
        if image_input:
            if os.path.exists(image_input):
                image_path = image_input
                logger.success(f"Image found: {image_path}")
            else:
                logger.warning(f"Image not found: {image_input}")
                logger.info("Proceeding without visual context.")
    
    return user_input, image_path

def get_user_instruction():
    """Backward compatibility function - returns only the instruction."""
    instruction, _ = get_user_input()
    return instruction

async def main(user_instruction=None, image_path=None):
    """Main application with LangSmith integration and visual context analysis."""
    
    logger.header("🏗️  REVIT AI ASSISTANT WITH LANGSMITH & VISUAL CONTEXT")
    logger.start_session()
    
    # Initialize method search engine
    logger.step(1, "Initializing Method Search Engine")
    initialize_method_search()
    
    # Get user instruction and image path if not provided
    if user_instruction is None:
        user_instruction, image_path = get_user_input()
    
    # Initialize visual context analyzer if enabled
    visual_context = None
    visual_similarity = 0.0
    if VISUAL_CONTEXT_ENABLED and image_path:
        logger.step(2, "Initializing Visual Context Analysis")
        try:
            analyzer = get_global_analyzer(device_choice=CLIP_DEVICE_CHOICE)
            visual_context, visual_similarity = analyzer.analyze_visual_context(image_path, user_instruction)
            
            logger.success(f"Visual analysis completed - Similarity: {visual_similarity:.3f}")
            logger.info(f"Visual Context: {visual_context}")
            
            # Get device info for logging
            device_info = analyzer.get_device_info()
            logger.info(f"CLIP Device: {device_info['actual_device']} (choice: {device_info['device_choice']})")
            
            # Get visual suggestions
            suggestions = analyzer.get_visual_suggestions(image_path, user_instruction)
            if suggestions:
                logger.info("Visual Suggestions:")
                for suggestion in suggestions:
                    logger.info(f"  💡 {suggestion}")
                    
        except Exception as e:
            logger.warning(f"Visual context analysis failed: {e}")
            logger.info("Proceeding with text-only analysis")
            visual_context = "Visual analysis unavailable. Proceeding with text-only planning."
            visual_similarity = 0.0
    else:
        logger.step(2, "Skipping Visual Context Analysis")
        if not VISUAL_CONTEXT_ENABLED:
            logger.info("Visual context analysis is disabled")
        elif not image_path:
            logger.info("No image provided for visual analysis")
        visual_context = "No visual context available. Proceeding with text-only planning."
    
    # Check LangSmith setup
    langsmith_enabled = setup_langsmith()
    
    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file")
    
    logger.success("Groq API key found")
    
    # Log MCP configuration
    if MCP_ENABLED:
        logger.success("MCP execution enabled")
        if ASYNC_EXECUTION:
            logger.info("Async execution mode enabled")
        else:
            logger.info("Sync execution mode enabled")
    else:
        logger.warning("MCP execution disabled - demo mode only")
    
    # Initialize the ChatGroq model
    logger.step(3, "Initializing AI Model")
    llm = ChatGroq(groq_api_key=api_key, model_name="llama3-70b-8192")
    logger.success("Llama3-70B model loaded")
    
    # Search for existing methods first
    logger.step(4, "Searching for Existing Methods")
    matching_methods = search_method_library(user_instruction)
    
    if matching_methods:
        logger.success(f"Found {len(matching_methods)} potentially relevant methods")
        for i, method in enumerate(matching_methods, 1):
            logger.info(f"  {i}. {method['method_name']}: {method['description']}")
    else:
        logger.info("No existing methods found for this instruction")
    
    # Create dynamic prompt with visual context
    logger.step(5, "Building Dynamic Prompt with Visual Context")
    
    # Get existing methods for the prompt
    existing_methods = create_method_library_description()
    
    # Add semantic search results to the prompt
    if matching_methods:
        semantic_results = "\n\n**Semantically Similar Methods Found:**\n"
        for i, method in enumerate(matching_methods, 1):
            semantic_results += f"{i}. **{method['method_name']}**: {method['description']}\n"
            if method.get('parameters'):
                semantic_results += f"   Parameters: {', '.join(method['parameters'])}\n"
            semantic_results += f"   Keywords: {', '.join(method.get('keywords', []))}\n\n"
    else:
        semantic_results = "\n\n**No semantically similar methods found.**\n"
    
    # Add visual context to the prompt
    visual_context_section = f"""
**Visual Context Analysis:**
{visual_context}

**Visual Similarity Score:** {visual_similarity:.3f}

**Visual-Aware Planning Guidelines:**
- If visual similarity > 0.7: Use visual context confidently in planning
- If visual similarity > 0.5: Consider visual elements in planning
- If visual similarity > 0.3: Use visual context as hints, verify assumptions
- If visual similarity < 0.3: Rely primarily on text instruction
"""
    
    prompt_template_str = f"""
You are an expert AI assistant that controls Autodesk Revit. Your goal is to break down complex user instructions into simple, executable Revit C# code steps.

**IMPORTANT**: First check if existing methods can handle the request, then create new methods only if needed.

{visual_context_section}

**Your Process:**
1. **Analyze** the user's instruction and visual context
2. **Check existing methods** - see if any can handle this request
3. **If method exists**: Use it with appropriate parameters
4. **If no method exists**: Create a new reusable method
5. **Break it down** into simple, sequential steps
6. **Convert each step** into executable C# code
7. **Output** a JSON with an array of C# code snippets

**Available Revit Objects:**
- `uiapp` - The UIApplication object (already available)
- `doc` - The current Document (uiapp.ActiveUIDocument.Document)
- `uidoc` - The current UIDocument (uiapp.ActiveUIDocument)
- `app` - The Application object (uiapp.Application)

**Common Revit Operations:**
- Create walls: `Wall.Create(doc, line, level, false)`
- Create floors: `Floor.Create(doc, curveArray, floorType, level)`
- Create doors/windows: `FamilyInstance.Create(doc, symbol, location, host)`
- Get levels: `FilteredElementCollector(doc).OfClass(typeof(Level))`
- Get categories: `FilteredElementCollector(doc).OfCategoryId(categoryId)`

**Interactive Operations (Getting Info from Revit):**
- Get selected elements: `uidoc.Selection.GetElementIds()`
- Get current view: `uidoc.ActiveView`
- Get element properties: `element.get_Parameter(BuiltInParameter.PARAM_NAME)`
- Get element location: `element.Location`
- Get element geometry: `element.get_Geometry(new Options())`
- Check element category: `element.Category.Id.IntegerValue == (int)BuiltInCategory.OST_Walls`

**Existing Methods Library:**
{existing_methods}

{semantic_results}

**Your output MUST be a single valid JSON object with this structure:**

**Option 1: Use existing method**
```json
{{{{
  "use_existing_method": {{{{
    "method_name": "create_window_on_wall",
    "parameters": {{{{
      "wall_position": "selected",
      "window_type": "default"
    }}}}
  }}}}
}}}}
```

**Option 2: Create new method**
```json
{{{{
  "create_new_method": {{{{
    "method_name": "create_window_on_wall",
    "description": "Creates a window on a selected wall",
    "keywords": ["window", "wall", "create", "door"],
    "parameters": ["wall_position", "window_type"],
    "csharp_code": "// Complete C# method code here\\npublic void CreateWindowOnWall(string wallPosition, string windowType) {{{{\\n  // Method implementation\\n}}}}"
  }}}}
}}}}
```

**Option 3: Direct execution (for simple tasks)**
```json
{{{{
  "task_breakdown": [
    {{{{
      "step_number": 1,
      "description": "Get selected wall information",
      "csharp_code": "// C# code to get selected wall\\nvar selectedIds = uidoc.Selection.GetElementIds();\\n// ... more code"
    }}}}
  ]
}}}}
```

**Rules:**
- **ALWAYS check existing methods first** before creating new ones
- **Reuse existing methods** when possible to avoid duplication
- **Create new methods** only when no suitable existing method exists
- **Make methods reusable** with parameters for flexibility
- **Include keywords** for better searchability
- **Use proper Revit API calls** and error handling
- **ALWAYS check for selected elements** when the instruction implies user interaction
- **ALWAYS validate element types** before using them
- **Consider semantic similarity** when choosing methods to reuse
- **Consider visual context** when available to enhance planning accuracy

**User Instruction:**
"{{instruction}}"

**Generate the JSON output now:**
"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template_str)
    chain = prompt | llm | StrOutputParser()
    
    # Count existing methods
    method_count = len(load_method_library())
    logger.success(f"Prompt built with {method_count} existing methods")
    
    # User instruction
    logger.step(6, "Processing User Instruction with Visual Context")
    logger.info(f"🎯 Instruction: '{user_instruction}'")
    if image_path:
        logger.info(f"📸 Image: {image_path}")
        logger.info(f"🔍 Visual Similarity: {visual_similarity:.3f}")
    
    # Run the chain (this will be traced in LangSmith if enabled)
    logger.step(7, "Executing AI Chain")
    
    start_time = time.time()
    result_str = chain.invoke({"instruction": user_instruction})
    execution_time = time.time() - start_time
    
    logger.info(f"⏱️  Execution time: {execution_time:.2f}s")
    logger.info("📊 Token usage and cost available in LangSmith dashboard")
    
    # Post-processing
    logger.step(8, "Processing AI Response")
    
    def extract_first_json(text):
        """Extract the first valid JSON object from a string using a stack-based approach."""
        start = text.find('{')
        if start == -1:
            return None
        stack = []
        for i in range(start, len(text)):
            if text[i] == '{':
                stack.append(i)
            elif text[i] == '}':
                stack.pop()
                if not stack:
                    try:
                        return json.loads(text[start:i+1])
                    except Exception as e:
                        logger.error(f"JSON decode error: {e}")
                        logger.info(f"Attempted to parse: {text[start:i+1][:200]}...")
                        return None
        return None

    def save_response_for_manual_review(response_text, instruction, attempt_number):
        """Save the full AI response for manual review."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"manual_review_{timestamp}_attempt_{attempt_number}.txt"
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Original Instruction: {instruction}\n")
                f.write(f"Attempt Number: {attempt_number}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("-" * 80 + "\n")
                f.write("Full AI Response:\n")
                f.write(response_text)
                f.write("\n" + "-" * 80 + "\n")
                f.write("Manual Review Instructions:\n")
                f.write("1. Look for valid JSON in the response\n")
                f.write("2. Extract and format the JSON manually\n")
                f.write("3. Save as method_library.json or execute manually\n")
            
            logger.warning(f"Full response saved to: {filename}")
            logger.info("Please review the file manually and extract valid JSON")
            return filename
        except Exception as e:
            logger.error(f"Failed to save response file: {e}")
            return None

    # Try to extract JSON, with retry if needed
    result_json = extract_first_json(result_str)
    if result_json is None:
        logger.warning("Initial JSON parsing failed, attempting retry...")
        retry_prompt = ChatPromptTemplate.from_template(
            "Your previous response was not valid JSON. Please try again and return ONLY a valid JSON object.\n\n"
            "Original instruction: {instruction}\n\n"
            "Your response must be valid JSON with no extra text, explanations, or markdown formatting."
        )
        retry_chain = retry_prompt | llm | StrOutputParser()
        
        retry_result = retry_chain.invoke({"instruction": user_instruction})
        logger.info("Retry completed - check LangSmith for details")
        
        result_json = extract_first_json(retry_result)

    # Fallback logic if both attempts fail
    if result_json is None:
        logger.error("Both JSON parsing attempts failed")
        
        # Save responses for manual review if enabled
        if SAVE_FAILED_RESPONSES:
            logger.warning("Saving responses for manual review...")
            save_response_for_manual_review(result_str, user_instruction, 1)
            save_response_for_manual_review(retry_result, user_instruction, 2)
        
        # Handle based on fallback mode
        if FALLBACK_MODE == "abort":
            logger.error("Fallback mode set to 'abort' - exiting")
            return
        elif FALLBACK_MODE == "manual":
            logger.warning("Fallback mode set to 'manual' - please review saved files")
            logger.info("Files saved for manual review. Please extract JSON and continue manually.")
            return
        else:  # auto mode (default)
            logger.warning("Fallback mode set to 'auto' - creating fallback method")
            
            # Create a simple fallback method based on the instruction
            fallback_method = {
                "method_name": f"fallback_{user_instruction.replace(' ', '_').lower()}",
                "description": f"Fallback method for: {user_instruction}",
                "keywords": user_instruction.lower().split(),
                "parameters": ["selected_element"],
                "csharp_code": f"// Fallback method for: {user_instruction}\\n// TODO: Implement based on manual review\\nlogger.Info(\"Fallback method called for: {user_instruction}\");"
            }
            
            save_method_to_library(fallback_method)
            logger.success("Fallback method created and saved to library")
            
            # Continue with the fallback method
            result_json = {"create_new_method": fallback_method}

    # Process results
    logger.step(9, "Analyzing Results")
    
    if "use_existing_method" in result_json:
        method_info = result_json["use_existing_method"]
        logger.info(f"Using existing method: {method_info['method_name']}")
        logger.info(f"   Parameters: {method_info.get('parameters', {})}")
        
        # Find the method in the library
        library = load_method_library()
        method = next((m for m in library if m['method_name'] == method_info['method_name']), None)
        
        if method:
            logger.success("Method found in library")
            logger.info(f"   Description: {method['description']}")
            logger.info(f"   Code: {method['csharp_code'][:100]}...")
        else:
            logger.error("Method not found in library")
            
    elif "create_new_method" in result_json:
        method_info = result_json["create_new_method"]
        logger.info(f"Creating new method: {method_info['method_name']}")
        logger.info(f"   Description: {method_info['description']}")
        logger.info(f"   Keywords: {method_info.get('keywords', [])}")
        logger.info(f"   Parameters: {method_info.get('parameters', [])}")
        
        # Save the new method to library
        save_method_to_library(method_info)
        
        # Show the generated code
        logger.info(f"   Code Preview: {method_info['csharp_code'][:200]}...")
        
    elif "task_breakdown" in result_json:
        task_breakdown = result_json["task_breakdown"]
        logger.info(f"Task breakdown created with {len(task_breakdown)} steps")
        
        # Visualize the task breakdown
        print_plan_visualization(task_breakdown)
        
        # Execute the task breakdown (if MCP server is available)
        logger.step(10, "Executing Task Breakdown")
        
        if MCP_ENABLED:
            try:
                # Create unified executor
                executor = create_executor(async_mode=ASYNC_EXECUTION)
                
                # Execute using unified interface
                results = await executor.execute(task_breakdown)
                
                # Log detailed results
                for result in results:
                    if isinstance(result, dict):
                        if result.get("success"):
                            logger.success(f"✅ {result['description']}")
                        else:
                            logger.error(f"❌ {result['description']}: {result.get('error', 'Unknown error')}")
                    else:
                        logger.error(f"❌ Unexpected result type: {type(result)}")
                        
            except Exception as e:
                logger.error(f"Execution failed: {str(e)}")
                logger.warning("Saving execution error for manual review...")
                
                # Save error details for manual review
                error_file = f"execution_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                try:
                    with open(error_file, "w") as f:
                        f.write(f"Execution Error: {str(e)}\n")
                        f.write(f"Task Breakdown: {json.dumps(task_breakdown, indent=2)}\n")
                        f.write(f"Timestamp: {datetime.now()}\n")
                    logger.info(f"Error details saved to: {error_file}")
                except Exception as save_error:
                    logger.error(f"Failed to save error file: {save_error}")
        else:
            logger.warning("MCP execution is disabled")
            logger.info("To enable execution, set MCP_ENABLED=true in your .env file")
            logger.info("For async execution, also set ASYNC_EXECUTION=true")
    
    else:
        logger.error("Unknown response format")
        logger.info(f"Response keys: {list(result_json.keys())}")
    
    logger.end_session()
    
    if langsmith_enabled:
        logger.info("\n🌐 View detailed traces and analytics at: https://smith.langchain.com/")
        logger.info("📊 You'll see:")
        logger.info("   - Chain execution flow")
        logger.info("   - Token usage graphs")
        logger.info("   - Prompt/response history")
        logger.info("   - Performance metrics")
        logger.info("   - Error tracking")

# --- Convenience Functions for Testing ---
async def run_with_instruction(instruction, image_path=None):
    """Run the main application with a specific instruction (for testing)."""
    return await main(user_instruction=instruction, image_path=image_path)

async def run_with_image(instruction, image_path):
    """Run the main application with a specific instruction and image (for testing)."""
    return await main(user_instruction=instruction, image_path=image_path)

async def run_interactive():
    """Run the main application with interactive user input."""
    return await main()

if __name__ == "__main__":
    # You can either run interactively or with a specific instruction
    # For testing specific instructions, uncomment and modify the line below:
    # asyncio.run(run_with_instruction("create a door on the selected wall"))
    
    # For testing with visual context, uncomment and modify the line below:
    # asyncio.run(run_with_image("create a window on the selected wall", "path/to/wall_image.jpg"))
    
    # For interactive mode (default):
    asyncio.run(run_interactive()) 