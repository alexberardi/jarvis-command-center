"""
Prompt Provider Factory for Jarvis Voice Assistant.

Discovers IJarvisPromptProvider implementations by scanning
app/core/prompt_providers/ recursively, then falls back to legacy
IModelInterface classes in app/core/models/ for backward compatibility.

Uses the same pkgutil.walk_packages + inspect.getmembers pattern as
ModelFactory but targets the prompt_providers package first.
"""

import importlib
import inspect
import logging
import os
import pkgutil
from typing import Any, Dict, List, Optional

from app.core.interfaces.ijarvis_prompt_provider import IJarvisPromptProvider

logger = logging.getLogger("uvicorn")


class PromptProviderFactory:
    """
    Factory for creating prompt provider instances.

    Discovery order:
    1. app/core/prompt_providers/ (new providers)
    2. app/core/models/ (legacy models via duck-typing)

    Matching: case-insensitive comparison on instance.name property.
    """

    @classmethod
    def create_provider(
        cls, provider_name: Optional[str] = None
    ) -> Optional[IJarvisPromptProvider]:
        """
        Create a prompt provider by name.

        Args:
            provider_name: Provider name to match (case-insensitive).
                If None, resolves via settings cascade.

        Returns:
            IJarvisPromptProvider instance, or None if not found
            (caller should fall back to ModelFactory).
        """
        if provider_name is None:
            provider_name = cls._resolve_provider_name()

        logger.info("PromptProviderFactory: looking for '%s'", provider_name)

        # Search prompt_providers package
        provider = cls._scan_prompt_providers(provider_name)
        if provider is not None:
            logger.info(
                "PromptProviderFactory: found provider %s (%s)",
                provider.name,
                provider.__class__.__name__,
            )
            return provider

        logger.info(
            "PromptProviderFactory: '%s' not found in prompt_providers",
            provider_name,
        )
        return None

    @classmethod
    def _scan_prompt_providers(cls, target_name: str) -> Optional[IJarvisPromptProvider]:
        """
        Scan app/core/prompt_providers/ for a matching provider.

        Args:
            target_name: Name to match (case-insensitive).

        Returns:
            Matching provider instance, or None.
        """
        providers_dir = os.path.join(os.path.dirname(__file__), "prompt_providers")
        if not os.path.exists(providers_dir):
            return None

        prefix = "app.core.prompt_providers."
        upper_target = target_name.upper()

        for _finder, module_name, _ispkg in pkgutil.walk_packages(
            path=[providers_dir], prefix=prefix
        ):
            try:
                imported = importlib.import_module(module_name)
            except Exception as e:
                logger.debug("Failed to import %s: %s", module_name, e)
                continue

            for _class_name, klass in inspect.getmembers(imported, inspect.isclass):
                if not (issubclass(klass, IJarvisPromptProvider) and klass is not IJarvisPromptProvider):
                    continue
                try:
                    instance = klass()
                    if instance.name.upper() == upper_target:
                        return instance
                except Exception as e:
                    logger.debug("Failed to instantiate %s: %s", klass.__name__, e)

        return None

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """
        List all discoverable prompt provider names.

        Returns:
            Sorted list of provider name strings.
        """
        names: list[str] = []
        providers_dir = os.path.join(os.path.dirname(__file__), "prompt_providers")
        if not os.path.exists(providers_dir):
            return names

        prefix = "app.core.prompt_providers."

        for _finder, module_name, _ispkg in pkgutil.walk_packages(
            path=[providers_dir], prefix=prefix
        ):
            try:
                imported = importlib.import_module(module_name)
            except (ImportError, AttributeError):
                continue

            for _class_name, klass in inspect.getmembers(imported, inspect.isclass):
                if not (issubclass(klass, IJarvisPromptProvider) and klass is not IJarvisPromptProvider):
                    continue
                try:
                    instance = klass()
                    if instance.name not in names:
                        names.append(instance.name)
                except (TypeError, ValueError):
                    pass

        return sorted(names)

    @classmethod
    def get_provider_info(cls, provider_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific provider.

        Args:
            provider_name: Name of the provider (case-insensitive).

        Returns:
            Provider info dict, or None if not found.
        """
        provider = cls.create_provider(provider_name)
        if provider is None:
            return None
        return {
            "name": provider.name,
            "class": provider.__class__.__name__,
            "module": provider.__class__.__module__,
            "capabilities": provider.get_capabilities(),
        }

    @staticmethod
    def _resolve_provider_name() -> str:
        """
        Resolve provider name from settings cascade.

        Same cascade as ModelFactory._resolve_model_name:
        1. Database setting ``llm.interface``
        2. JARVIS_MODEL_INTERFACE env var
        3. Hardcoded fallback
        """
        try:
            # Lazy import to avoid circular dependency with settings_service
            from app.services.settings_service import get_settings_service
            settings = get_settings_service()
            name = settings.get("llm.interface")
            if name:
                logger.info("PromptProviderFactory: resolved '%s' from settings", name)
                return name
        except Exception as e:
            logger.warning(
                "PromptProviderFactory: could not read llm.interface: %s", e
            )

        fallback = os.getenv("JARVIS_MODEL_INTERFACE", "JarvisToolModel")
        logger.info("PromptProviderFactory: using fallback '%s'", fallback)
        return fallback
