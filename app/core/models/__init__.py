"""
Model implementations for Jarvis Voice Assistant.

This package contains different model implementations that can be used
with the Jarvis system. Each model implements the IModelInterface and
can have different inference strategies.

Available Models:
- BaseModel: Reference implementation showing full system complexity
- JarvisLlama3B: Fine-tuned Llama 3.2 3B with single-shot inference
- JarvisParallelLlama3B: Llama 3.2 3B with parallel double-shot inference
"""

from app.core.models.base_model import BaseModel
from app.core.models.jarvis_llama3b import JarvisLlama3B
from app.core.models.jarvis_parallel_llama3b import JarvisParallelLlama3B

__all__ = [
    "BaseModel",
    "JarvisLlama3B",
    "JarvisParallelLlama3B"
]
