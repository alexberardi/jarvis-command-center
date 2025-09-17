"""
Custom model implementations.

This directory is for custom model implementations that extend or replace
the built-in Jarvis models. Custom models should implement IModelInterface
and can be loaded automatically by the ModelFactory.

To create a custom model:
1. Create a new .py file in this directory
2. Implement IModelInterface
3. Set JARVIS_MODEL_INTERFACE environment variable to your model's name

Example:
    class MyCustomModel(IModelInterface):
        @property
        def name(self) -> str:
            return "MyCustomModel"
        
        # Implement other required methods...
"""
