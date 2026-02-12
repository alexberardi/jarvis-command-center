"""
Model Factory for Jarvis Voice Assistant.

This factory provides a clean way to instantiate different model implementations
based on environment configuration. It dynamically discovers and loads models
from the models directory.
"""

import os
import logging
import importlib
import inspect
import pkgutil
from typing import Optional, List, Dict, Any

from app.core.interfaces.imodel_interface import IModelInterface

logger = logging.getLogger("uvicorn")


class ModelFactory:
    """
    Factory for creating model implementations.
    
    Discovers models by scanning:
    - app/core/models/ (built-in models)
    - app/core/models/custom/ (custom models)
    
    Matches models by their `name` property against JARVIS_MODEL_INTERFACE env variable.
    """
    
    @classmethod
    def create_model(cls, model_name: Optional[str] = None) -> IModelInterface:
        """
        Create a model instance based on configuration.
        
        Args:
            model_name: Specific model name to create. If None, uses environment variable.
            
        Returns:
            Model implementation instance
            
        Raises:
            ValueError: If model name is not found
        """
        # Determine model name from env or parameter
        if model_name is None:
            model_name = os.getenv("JARVIS_MODEL_INTERFACE", "JarvisToolModel")
        
        logger.info(f"🏭 Creating model: {model_name}")
        
        # Search for model implementation
        base_paths = [
            os.path.join(os.path.dirname(__file__), "models"),
            os.path.join(os.path.dirname(__file__), "models", "custom")
        ]
        
        prefix = "app.core.models."
        
        for path in base_paths:
            if not os.path.exists(path):
                continue
            
            for finder, module_name, ispkg in pkgutil.walk_packages(path=[path], prefix=prefix):
                try:
                    imported_module = importlib.import_module(module_name)
                except Exception as e:
                    logger.debug(f"Failed to import module {module_name}: {e}")
                    continue
                
                # Look for classes that implement IModelInterface
                for class_name, cls in inspect.getmembers(imported_module, inspect.isclass):
                    if issubclass(cls, IModelInterface) and cls is not IModelInterface:
                        try:
                            instance = cls()
                            # Match by instance.name property (case-insensitive)
                            if instance.name.upper() == model_name.upper():
                                logger.info(f"✅ Found model: {cls.__name__} (name={instance.name})")
                                return instance
                        except Exception as e:
                            logger.warning(f"Failed to instantiate {cls.__name__}: {e}")
                            continue
        
        # Model not found
        available = cls.get_available_models()
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {available}"
        )
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """
        Get list of all available model names by scanning directories.
        
        Returns:
            List of model names (from their `name` property)
        """
        models = []
        
        base_paths = [
            os.path.join(os.path.dirname(__file__), "models"),
            os.path.join(os.path.dirname(__file__), "models", "custom")
        ]
        
        prefix = "app.core.models."
        
        for path in base_paths:
            if not os.path.exists(path):
                continue
            
            for finder, module_name, ispkg in pkgutil.walk_packages(path=[path], prefix=prefix):
                try:
                    imported_module = importlib.import_module(module_name)
                    
                    for class_name, cls in inspect.getmembers(imported_module, inspect.isclass):
                        if issubclass(cls, IModelInterface) and cls is not IModelInterface:
                            try:
                                instance = cls()
                                if instance.name not in models:
                                    models.append(instance.name)
                            except (TypeError, ValueError) as e:
                                pass

                except (ImportError, AttributeError) as e:
                    continue
        
        return sorted(models)
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dict or None if not found
        """
        try:
            model = cls.create_model(model_name)
            return {
                "name": model.name,
                "class": model.__class__.__name__,
                "capabilities": model.get_capabilities(),
                "module": model.__class__.__module__
            }
        except Exception as e:
            logger.warning(f"Failed to get info for model {model_name}: {e}")
            return None
