#!/usr/bin/env python3
"""
Visual Context Analysis using CLIP for Revit AI Assistant.
This module provides visual understanding capabilities to enhance the AI planning process.
"""

import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Conditional imports with proper error handling
try:
    import torch
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel
    from typing import Tuple, Optional, Dict, Any
    CLIP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"CLIP dependencies not available: {e}")
    logger.info("Install with: pip install torch transformers Pillow")
    CLIP_AVAILABLE = False
    # Create dummy types for type hints
    Tuple = type((None,))
    Optional = type(None)
    Dict = type({})
    Any = type(None)

class VisualContextAnalyzer:
    """CLIP-based visual context analyzer for Revit instructions."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device_choice: str = "auto"):
        """
        Initialize the CLIP model for visual context analysis.
        
        Args:
            model_name: CLIP model to use (default: openai/clip-vit-base-patch32)
            device_choice: Device to use - "auto", "cpu", "gpu", or "cuda"
        """
        if not CLIP_AVAILABLE:
            logger.error("CLIP dependencies not available. Cannot initialize analyzer.")
            self.model = None
            self.processor = None
            self.device = "cpu"
            self.device_choice = device_choice
            return
            
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device_choice = device_choice
        self.device = self._determine_device(device_choice)
        self._initialize_model()
    
    def _determine_device(self, device_choice: str) -> str:
        """
        Determine the best device to use based on user choice and availability.
        
        Args:
            device_choice: User's device preference
            
        Returns:
            Device string to use
        """
        if device_choice == "cpu":
            logger.info("Using CPU as requested")
            return "cpu"
        
        elif device_choice == "gpu" or device_choice == "cuda":
            if torch.cuda.is_available():
                logger.info("Using CUDA GPU as requested")
                return "cuda"
            else:
                logger.warning("GPU requested but CUDA not available. Falling back to CPU.")
                return "cpu"
        
        elif device_choice == "auto":
            if torch.cuda.is_available():
                logger.info("Auto-detected CUDA GPU available")
                return "cuda"
            else:
                logger.info("No CUDA GPU available, using CPU")
                return "cpu"
        
        else:
            logger.warning(f"Unknown device choice '{device_choice}'. Using auto-detection.")
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
    
    def _initialize_model(self):
        """Initialize the CLIP model and processor."""
        if not CLIP_AVAILABLE:
            logger.error("Cannot initialize model - CLIP not available")
            return
            
        try:
            logger.info(f"Initializing CLIP model: {self.model_name}")
            logger.info(f"Device choice: {self.device_choice}")
            logger.info(f"Using device: {self.device}")
            
            # Show device info
            if self.device == "cuda":
                logger.info(f"GPU: {torch.cuda.get_device_name()}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            
            # Move model to appropriate device
            self.model = self.model.to(self.device)
            
            logger.info("CLIP model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CLIP model: {e}")
            self.model = None
            self.processor = None
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get information about the current device setup.
        
        Returns:
            Dictionary with device information
        """
        info = {
            "device_choice": self.device_choice,
            "actual_device": self.device,
            "clip_available": CLIP_AVAILABLE,
            "model_initialized": self.model is not None
        }
        
        if CLIP_AVAILABLE:
            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name()
                info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return info
    
    def analyze_visual_context(self, image_path: str, user_instruction: str) -> Tuple[str, float]:
        """
        Analyze visual context and compute similarity with user instruction.
        
        Args:
            image_path: Path to the image file
            user_instruction: User's text instruction
            
        Returns:
            Tuple of (visual_context_description, similarity_score)
        """
        if not CLIP_AVAILABLE:
            logger.warning("CLIP not available, returning default context")
            return "Visual analysis unavailable. Proceeding with text-only planning.", 0.0
            
        if not self.model or not self.processor:
            logger.warning("CLIP model not initialized, returning default context")
            return "Visual analysis unavailable. Proceeding with text-only planning.", 0.0
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            # Prepare inputs
            inputs = self.processor(
                text=[user_instruction], 
                images=image, 
                return_tensors="pt", 
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Calculate similarity
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
            # Normalize embeddings for cosine similarity
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            similarity_score = torch.cosine_similarity(image_embeds, text_embeds)[0].item()
            
            # Generate context description based on similarity
            visual_context = self._generate_context_description(similarity_score, user_instruction)
            
            logger.info(f"Visual analysis completed - Similarity: {similarity_score:.3f}")
            
            return visual_context, similarity_score
            
        except Exception as e:
            logger.error(f"Visual analysis failed: {e}")
            return "Visual analysis failed. Proceeding with text-only planning.", 0.0
    
    def _generate_context_description(self, similarity_score: float, instruction: str) -> str:
        """
        Generate a descriptive context based on similarity score.
        
        Args:
            similarity_score: CLIP similarity score (0-1)
            instruction: User instruction for context
            
        Returns:
            Descriptive context string
        """
        if similarity_score > 0.7:
            return f"The image strongly matches the instruction '{instruction}'. The visual context confirms the user's intent. Proceed with confidence using visual-aware planning."
        
        elif similarity_score > 0.5:
            return f"The image moderately matches the instruction '{instruction}'. The visual context provides some relevant information. Consider visual elements in planning."
        
        elif similarity_score > 0.3:
            return f"The image has some relevance to the instruction '{instruction}'. The visual context may provide useful hints. Proceed with caution and verify assumptions."
        
        else:
            return f"The image has low relevance to the instruction '{instruction}'. The visual context may not be helpful. Rely primarily on the text instruction for planning."
    
    def analyze_multiple_images(self, image_paths: list, user_instruction: str) -> Dict[str, Any]:
        """
        Analyze multiple images and provide aggregated context.
        
        Args:
            image_paths: List of image file paths
            user_instruction: User's text instruction
            
        Returns:
            Dictionary with aggregated analysis results
        """
        if not CLIP_AVAILABLE:
            return {
                "visual_context": "CLIP not available. Proceeding with text-only planning.",
                "similarity_score": 0.0,
                "image_count": 0,
                "best_match": None
            }
            
        if not image_paths:
            return {
                "visual_context": "No images provided. Proceeding with text-only planning.",
                "similarity_score": 0.0,
                "image_count": 0,
                "best_match": None
            }
        
        results = []
        for i, image_path in enumerate(image_paths):
            try:
                context, score = self.analyze_visual_context(image_path, user_instruction)
                results.append({
                    "image_path": image_path,
                    "context": context,
                    "similarity_score": score
                })
            except Exception as e:
                logger.error(f"Failed to analyze image {image_path}: {e}")
                results.append({
                    "image_path": image_path,
                    "context": "Analysis failed",
                    "similarity_score": 0.0
                })
        
        # Find best match
        best_match = max(results, key=lambda x: x["similarity_score"])
        avg_score = sum(r["similarity_score"] for r in results) / len(results)
        
        # Generate aggregated context
        if avg_score > 0.5:
            aggregated_context = f"Multiple images analyzed ({len(results)} total). Average similarity: {avg_score:.3f}. Best match: {best_match['image_path']}. Visual context supports the instruction."
        else:
            aggregated_context = f"Multiple images analyzed ({len(results)} total). Average similarity: {avg_score:.3f}. Visual context may not be strongly relevant."
        
        return {
            "visual_context": aggregated_context,
            "similarity_score": avg_score,
            "image_count": len(results),
            "best_match": best_match,
            "all_results": results
        }
    
    def get_visual_suggestions(self, image_path: str, instruction: str) -> list:
        """
        Generate visual suggestions based on image analysis.
        
        Args:
            image_path: Path to the image file
            instruction: User instruction
            
        Returns:
            List of visual suggestions
        """
        if not CLIP_AVAILABLE:
            return ["CLIP not available. Install with: pip install torch transformers Pillow"]
            
        context, score = self.analyze_visual_context(image_path, instruction)
        
        suggestions = []
        
        if score > 0.6:
            suggestions.append("The image strongly supports your instruction. Consider using visual references in the generated code.")
            suggestions.append("You may want to extract specific measurements or positions from the image.")
        
        if "wall" in instruction.lower() and score > 0.4:
            suggestions.append("The image appears to show wall elements. Consider wall type, height, and material properties.")
        
        if "window" in instruction.lower() or "door" in instruction.lower():
            if score > 0.4:
                suggestions.append("The image shows openings. Consider size, type, and positioning based on visual context.")
        
        if "floor" in instruction.lower() and score > 0.4:
            suggestions.append("The image shows floor elements. Consider floor type, boundaries, and level information.")
        
        return suggestions

# Convenience function for easy integration
def analyze_visual_context(image_path: str, user_instruction: str, device_choice: str = "auto") -> Tuple[str, float]:
    """
    Convenience function for quick visual context analysis.
    
    Args:
        image_path: Path to the image file
        user_instruction: User's text instruction
        device_choice: Device to use - "auto", "cpu", "gpu", or "cuda"
        
    Returns:
        Tuple of (visual_context_description, similarity_score)
    """
    analyzer = VisualContextAnalyzer(device_choice=device_choice)
    return analyzer.analyze_visual_context(image_path, user_instruction)

# Global analyzer instance for reuse
_global_analyzer = None

def get_global_analyzer(device_choice: str = "auto") -> VisualContextAnalyzer:
    """Get or create a global analyzer instance."""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = VisualContextAnalyzer(device_choice=device_choice)
    return _global_analyzer

def reset_global_analyzer(device_choice: str = "auto") -> VisualContextAnalyzer:
    """Reset the global analyzer with new device choice."""
    global _global_analyzer
    _global_analyzer = VisualContextAnalyzer(device_choice=device_choice)
    return _global_analyzer 