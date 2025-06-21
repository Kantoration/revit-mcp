#!/usr/bin/env python3
"""
Setup script to enable LangSmith visualization for LangChain.
This will allow you to see your chains, prompts, and executions in a web interface.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_langsmith():
    """Set up LangSmith for visualization."""
    
    # Check if LANGCHAIN_API_KEY is set
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("⚠️  LANGCHAIN_API_KEY not found in .env file")
        print("To enable LangSmith visualization:")
        print("1. Go to https://smith.langchain.com/")
        print("2. Sign up for a free account")
        print("3. Get your API key from the settings")
        print("4. Add to your .env file:")
        print("   LANGCHAIN_API_KEY=your_api_key_here")
        print("5. Add to your .env file:")
        print("   LANGCHAIN_TRACING_V2=true")
        print("6. Add to your .env file:")
        print("   LANGCHAIN_PROJECT=your_project_name")
        return False
    
    print("✅ LangSmith is configured!")
    print("🌐 View your traces at: https://smith.langchain.com/")
    print("📊 You'll see:")
    print("   - Chain execution flow")
    print("   - Token usage")
    print("   - Prompt inputs/outputs")
    print("   - Tool calls")
    print("   - Performance metrics")
    return True

if __name__ == "__main__":
    setup_langsmith() 