#!/usr/bin/env python3
"""
Test script to demonstrate semantic search functionality.
"""

import json
import asyncio
from main import search_engine, logger, initialize_method_search

def create_sample_methods():
    """Create sample methods for testing semantic search."""
    sample_methods = [
        {
            "method_name": "create_window_on_wall",
            "description": "Creates a window on a selected wall at a specified position",
            "keywords": ["window", "wall", "create", "opening", "insert"],
            "parameters": ["wall_position", "window_type", "height"],
            "csharp_code": "public void CreateWindowOnWall(string wallPosition, string windowType, double height) { /* implementation */ }"
        },
        {
            "method_name": "create_door_on_wall",
            "description": "Creates a door on a selected wall with specified parameters",
            "keywords": ["door", "wall", "create", "opening", "insert"],
            "parameters": ["wall_position", "door_type", "height"],
            "csharp_code": "public void CreateDoorOnWall(string wallPosition, string doorType, double height) { /* implementation */ }"
        },
        {
            "method_name": "get_element_information",
            "description": "Retrieves detailed information about selected elements",
            "keywords": ["get", "information", "selected", "element", "properties"],
            "parameters": ["element_ids"],
            "csharp_code": "public Dictionary<string, object> GetElementInformation(List<ElementId> elementIds) { /* implementation */ }"
        },
        {
            "method_name": "create_floor_at_level",
            "description": "Creates a floor at a specified level with given boundaries",
            "keywords": ["floor", "level", "create", "boundary", "geometry"],
            "parameters": ["level_name", "boundary_curves", "floor_type"],
            "csharp_code": "public void CreateFloorAtLevel(string levelName, CurveArray boundaryCurves, FloorType floorType) { /* implementation */ }"
        },
        {
            "method_name": "create_wall_between_points",
            "description": "Creates a wall between two specified points",
            "keywords": ["wall", "create", "points", "line", "geometry"],
            "parameters": ["start_point", "end_point", "wall_type", "level"],
            "csharp_code": "public void CreateWallBetweenPoints(XYZ startPoint, XYZ endPoint, WallType wallType, Level level) { /* implementation */ }"
        }
    ]
    
    # Save to method library
    with open("method_library.json", "w") as f:
        json.dump(sample_methods, f, indent=2)
    
    logger.success(f"Created {len(sample_methods)} sample methods")
    return sample_methods

def test_semantic_search():
    """Test the semantic search functionality with various queries."""
    
    # Initialize search engine
    logger.header("🧪 TESTING SEMANTIC SEARCH")
    initialize_method_search()
    
    # Test queries
    test_queries = [
        "create a window on the selected wall",
        "add a door to the wall",
        "get information about selected elements",
        "create a floor at level 1",
        "build a wall between two points",
        "insert an opening in the wall",
        "retrieve element properties",
        "make a new floor",
        "construct a wall",
        "find element details"
    ]
    
    logger.info("Testing semantic search with various queries...")
    logger.info("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n🔍 Test {i}: '{query}'")
        logger.info("-" * 40)
        
        # Perform search
        results = search_engine.search_semantic(query, top_k=2)
        
        if results:
            for j, method in enumerate(results, 1):
                logger.success(f"  {j}. {method['method_name']}")
                logger.info(f"     Description: {method['description']}")
                logger.info(f"     Keywords: {', '.join(method.get('keywords', []))}")
        else:
            logger.warning("  No relevant methods found")
    
    logger.info("\n" + "=" * 60)
    logger.success("Semantic search testing completed!")

async def main():
    """Main test function."""
    logger.start_session()
    
    # Create sample methods if they don't exist
    try:
        with open("method_library.json", "r") as f:
            existing_methods = json.load(f)
        logger.info(f"Found {len(existing_methods)} existing methods")
    except FileNotFoundError:
        logger.info("No method library found, creating sample methods...")
        create_sample_methods()
    
    # Test semantic search
    test_semantic_search()
    
    logger.end_session()

if __name__ == "__main__":
    asyncio.run(main()) 