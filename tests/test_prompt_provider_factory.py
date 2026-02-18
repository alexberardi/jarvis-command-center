"""
Tests for PromptProviderFactory.

Verifies:
- Factory discovers providers from app/core/prompt_providers/
- Case-insensitive name matching
- Returns None for unknown names (caller falls back to ModelFactory)
- get_available_providers lists discovered providers
- get_provider_info returns capabilities
"""

import pytest

from app.core.prompt_provider_factory import PromptProviderFactory
from app.core.interfaces.ijarvis_prompt_provider import IJarvisPromptProvider


class TestPromptProviderFactoryDiscovery:
    """Test provider discovery from prompt_providers package."""

    def test_finds_llama_small_untrained(self):
        """Factory should discover LlamaSmallUntrained provider."""
        provider = PromptProviderFactory.create_provider("LlamaSmallUntrained")
        assert provider is not None
        assert provider.name == "LlamaSmallUntrained"
        assert isinstance(provider, IJarvisPromptProvider)

    def test_case_insensitive_matching(self):
        """Factory should match names case-insensitively."""
        provider = PromptProviderFactory.create_provider("llamasmalluntrained")
        assert provider is not None
        assert provider.name == "LlamaSmallUntrained"

    def test_case_insensitive_uppercase(self):
        """Factory should match all-uppercase names."""
        provider = PromptProviderFactory.create_provider("LLAMASMALLUNTRAINED")
        assert provider is not None
        assert provider.name == "LlamaSmallUntrained"

    def test_returns_none_for_unknown_name(self):
        """Factory should return None for names not in prompt_providers."""
        provider = PromptProviderFactory.create_provider("NonExistentProvider")
        assert provider is None

    def test_returns_none_for_legacy_model_name(self):
        """Factory should return None for legacy model names (not in prompt_providers)."""
        provider = PromptProviderFactory.create_provider("JarvisToolModel")
        assert provider is None


class TestPromptProviderFactoryListing:
    """Test provider listing and info."""

    def test_get_available_providers_includes_llama(self):
        """Available providers should include LlamaSmallUntrained."""
        providers = PromptProviderFactory.get_available_providers()
        assert "LlamaSmallUntrained" in providers

    def test_get_available_providers_returns_sorted(self):
        """Available providers should be sorted."""
        providers = PromptProviderFactory.get_available_providers()
        assert providers == sorted(providers)

    def test_get_provider_info_success(self):
        """get_provider_info should return capabilities for known provider."""
        info = PromptProviderFactory.get_provider_info("LlamaSmallUntrained")
        assert info is not None
        assert info["name"] == "LlamaSmallUntrained"
        assert info["class"] == "LlamaSmallUntrained"
        assert "capabilities" in info
        assert info["capabilities"]["model_family"] == "llama"
        assert info["capabilities"]["size_tier"] == "small"
        assert info["capabilities"]["training_tier"] == "untrained"

    def test_get_provider_info_unknown(self):
        """get_provider_info should return None for unknown provider."""
        info = PromptProviderFactory.get_provider_info("DoesNotExist")
        assert info is None
