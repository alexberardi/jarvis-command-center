"""
Prompt Provider Factory for Jarvis Voice Assistant.

Discovers IJarvisPromptProvider implementations by scanning two roots:
  1. app/core/prompt_providers/ — built-in providers shipped with the image
  2. app/core/prompt_providers_custom/ — user-installed providers under a
     named docker volume so they survive image upgrades

Built-ins are scanned first; on the first match create_provider returns.
For get_available_providers, both roots are walked and results are
deduplicated by provider name.
"""

import importlib
import inspect
import logging
import os
import pkgutil
from typing import Any, Dict, List, Optional

from app.core.interfaces.ijarvis_prompt_provider import IJarvisPromptProvider

logger = logging.getLogger("uvicorn")

# (relative-dir-name, importlib-prefix) — scanned in this order.
_PROVIDER_ROOTS: tuple[tuple[str, str], ...] = (
    ("prompt_providers", "app.core.prompt_providers."),
    ("prompt_providers_custom", "app.core.prompt_providers_custom."),
)


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
        Scan both prompt_providers roots for a matching provider.

        Built-ins are scanned first; first match wins.

        Args:
            target_name: Name to match (case-insensitive).

        Returns:
            Matching provider instance, or None.
        """
        upper_target = target_name.upper()
        core_dir = os.path.dirname(__file__)

        for dir_name, prefix in _PROVIDER_ROOTS:
            providers_dir = os.path.join(core_dir, dir_name)
            if not os.path.exists(providers_dir):
                continue

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
        List all discoverable prompt provider names across both roots.

        Returns:
            Sorted list of provider name strings, deduplicated by name.
        """
        names: list[str] = []
        core_dir = os.path.dirname(__file__)

        for dir_name, prefix in _PROVIDER_ROOTS:
            providers_dir = os.path.join(core_dir, dir_name)
            if not os.path.exists(providers_dir):
                continue

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

        fallback = "JarvisToolModel"
        logger.info("PromptProviderFactory: using fallback '%s'", fallback)
        return fallback
