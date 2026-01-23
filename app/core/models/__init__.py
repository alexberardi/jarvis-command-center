"""
Model implementations for Jarvis Voice Assistant.

This package contains different model implementations that can be used
with the Jarvis system. Each model implements the IModelInterface and
can have different inference strategies.

Available Models:
- JarvisToolModel: Tool-based model for prompt-driven tool calling
- JarvisAdapterModel: Slim prompt model for adapter-tuned usage
"""

from app.core.models.jarvis_tool_model import JarvisToolModel
from app.core.models.jarvis_adapter_model import JarvisAdapterModel

__all__ = [
    "JarvisToolModel",
    "JarvisAdapterModel"
]
