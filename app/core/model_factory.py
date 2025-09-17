"""
Model Factory for Jarvis Voice Assistant.

This factory provides a clean way to instantiate different model implementations
based on environment configuration. It supports both built-in models and
custom implementations that can be dynamically loaded.
"""

import os
import logging
import importlib
import inspect
import pkgutil
from typing import Optional, List, Dict, Any

from app.core.interfaces.imodel_interface import IModelInterface
from app.core.models.base_model import BaseModel
from app.core.models.jarvis_llama3b import JarvisLlama3B
from app.core.models.jarvis_parallel_llama3b import JarvisParallelLlama3B

logger = logging.getLogger("uvicorn")

class ModelFactory:
    """
    Factory for creating model implementations.
    
    Supports:
    1. Built-in models (BaseModel, JarvisLlama3B, etc.)
    2. Custom models loaded dynamically from app/core/models/custom/
    3. Environment variable configuration
    """
    
    # Built-in model registry
    BUILTIN_MODELS = {
        "BASE_MODEL": BaseModel,
        "JarvisLlama3B": JarvisLlama3B,
        "JarvisParallelLlama3B": JarvisParallelLlama3B,
    }
    
    @classmethod
    def create_model(self, model_name: Optional[str] = None) -> IModelInterface:
        """
        Create a model instance based on configuration.
        
        Args:
            model_name: Specific model name to create. If None, uses environment variable.
            
        Returns:
            Model implementation instance
            
        Raises:
            ValueError: If model name is not found
            ImportError: If custom model cannot be loaded
        """
        # Determine model name
        if model_name is None:
            model_name = os.getenv("JARVIS_MODEL_INTERFACE", "BASE_MODEL")
        
        logger.info(f"🏭 Creating model: {model_name}")
        
        # Try built-in models first
        if model_name in self.BUILTIN_MODELS:
            model_class = self.BUILTIN_MODELS[model_name]
            logger.info(f"✅ Using built-in model: {model_class.__name__}")
            return model_class()
        
        # Try to load custom model
        custom_model = self._load_custom_model(model_name)
        if custom_model:
            logger.info(f"✅ Using custom model: {custom_model.__class__.__name__}")
            return custom_model
        
        # Model not found
        available_models = list(self.BUILTIN_MODELS.keys())
        custom_models = self._get_available_custom_models()
        available_models.extend(custom_models)
        
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {available_models}"
        )
    
    @classmethod
    def _load_custom_model(cls, model_name: str) -> Optional[IModelInterface]:
        """
        Load a custom model implementation.
        
        Searches for custom models in:
        - app/core/models/custom/
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Model instance if found, None otherwise
        """
        search_paths = [
            os.path.join(os.path.dirname(__file__), "custom"),
        ]
        
        for search_path in search_paths:
            if not os.path.exists(search_path):
                continue
                
            # Walk through all modules in the search path
            prefix = "app.core.models.custom."
            
            try:
                for finder, module_name, ispkg in pkgutil.walk_packages(
                    path=[search_path], 
                    prefix=prefix
                ):
                    try:
                        imported_module = importlib.import_module(module_name)
                    except Exception as e:
                        logger.warning(f"Failed to import custom model module {module_name}: {e}")
                        continue
                    
                    # Look for classes that implement IModelInterface
                    for name, cls in inspect.getmembers(imported_module, inspect.isclass):
                        if (issubclass(cls, IModelInterface) and 
                            cls is not IModelInterface and
                            hasattr(cls, 'name')):
                            
                            try:
                                instance = cls()
                                if instance.name == model_name:
                                    logger.info(f"Found custom model: {cls.__name__}")
                                    return instance
                            except Exception as e:
                                logger.warning(f"Failed to instantiate custom model {cls.__name__}: {e}")
                                continue
                                
            except Exception as e:
                logger.warning(f"Error searching for custom models in {search_path}: {e}")
                continue
        
        return None
    
    @classmethod
    def _get_available_custom_models(cls) -> List[str]:
        """Get list of available custom model names."""
        custom_models = []
        search_paths = [
            os.path.join(os.path.dirname(__file__), "custom"),
        ]
        
        for search_path in search_paths:
            if not os.path.exists(search_path):
                continue
                
            prefix = "app.core.models.custom."
            
            try:
                for finder, module_name, ispkg in pkgutil.walk_packages(
                    path=[search_path], 
                    prefix=prefix
                ):
                    try:
                        imported_module = importlib.import_module(module_name)
                        
                        for name, cls in inspect.getmembers(imported_module, inspect.isclass):
                            if (issubclass(cls, IModelInterface) and 
                                cls is not IModelInterface and
                                hasattr(cls, 'name')):
                                
                                try:
                                    instance = cls()
                                    custom_models.append(instance.name)
                                except:
                                    pass
                                    
                    except Exception:
                        continue
                        
            except Exception:
                continue
        
        return custom_models
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of all available model names."""
        models = list(cls.BUILTIN_MODELS.keys())
        models.extend(cls._get_available_custom_models())
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
                "is_builtin": model_name in cls.BUILTIN_MODELS
            }
        except Exception as e:
            logger.warning(f"Failed to get info for model {model_name}: {e}")
            return None
