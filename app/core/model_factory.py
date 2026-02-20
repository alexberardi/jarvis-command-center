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
    
    Matches models by their `name` property against the ``llm.interface`` database
    setting (cascade: DB → JARVIS_MODEL_INTERFACE env var → default).
    """

    @classmethod
    def create_model(cls, model_name: Optional[str] = None) -> IModelInterface:
        """
        Create a model instance based on configuration.

        Args:
            model_name: Specific model name to create. If None, reads from
                        database setting ``llm.interface`` (falls back to
                        JARVIS_MODEL_INTERFACE env var, then JarvisToolModel).

        Returns:
            Model implementation instance

        Raises:
            ValueError: If model name is not found
        """
        # Determine model name: explicit arg → DB setting → env var → default
        if model_name is None:
            model_name = ModelFactory._resolve_model_name()
        
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
                for class_name, klass in inspect.getmembers(imported_module, inspect.isclass):
                    if issubclass(klass, IModelInterface) and klass is not IModelInterface:
                        try:
                            instance = klass()
                            # Match by instance.name property (case-insensitive)
                            if instance.name.upper() == model_name.upper():
                                logger.info(f"✅ Found model: {klass.__name__} (name={instance.name})")
                                return instance
                        except Exception as e:
                            logger.warning(f"Failed to instantiate {klass.__name__}: {e}")
                            continue

        # Model not found
        available = ModelFactory.get_available_models()
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {available}"
        )
    
    @staticmethod
    def _resolve_model_name() -> str:
        """Resolve model name from database settings with env var fallback.

        Precedence (handled by settings service):
        1. Database value (user > node > household > system)
        2. JARVIS_MODEL_INTERFACE environment variable
        3. Definition default (JarvisAdapterModel)
        4. Hardcoded fallback (JarvisToolModel)
        """
        try:
            from app.services.settings_service import get_settings_service
            settings = get_settings_service()
            name = settings.get("llm.interface")
            if name:
                logger.info(f"🏭 Resolved model from settings: {name}")
                return name
        except Exception as e:
            logger.warning(f"🏭 Could not read llm.interface from settings: {e}")

        # Final fallback if settings service is unavailable
        fallback = "JarvisToolModel"
        logger.info(f"🏭 Using fallback for model: {fallback}")
        return fallback

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
