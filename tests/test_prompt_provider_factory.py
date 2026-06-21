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


class TestPromptProviderFactoryDualRoot:
    """Test that factory scans both ``prompt_providers/`` (built-ins) and
    ``prompt_providers_custom/`` (user-installed) roots (#8).

    Built-in regression tests use the real packaged providers.
    Custom-root tests mock pkgutil + importlib to inject stub providers
    without polluting the production discovery tree.
    """

    @pytest.fixture
    def mock_scan(self, monkeypatch):
        """Inject stub providers into either or both factory scan roots.

        Returns a ``register(root, module_name, provider_class)`` callable.
        ``root`` is ``"prompt_providers"`` or ``"prompt_providers_custom"``.
        """
        import types

        yielded_modules: dict[str, list[str]] = {
            "prompt_providers": [],
            "prompt_providers_custom": [],
        }
        fake_modules: dict[str, types.ModuleType] = {}

        def fake_walk_packages(path, prefix):
            if "prompt_providers_custom" in prefix:
                root = "prompt_providers_custom"
            else:
                root = "prompt_providers"
            for module_name in yielded_modules[root]:
                yield None, module_name, False

        def fake_import_module(name):
            if name in fake_modules:
                return fake_modules[name]
            raise ImportError(f"unknown stub module: {name}")

        monkeypatch.setattr(
            "app.core.prompt_provider_factory.pkgutil.walk_packages",
            fake_walk_packages,
        )
        monkeypatch.setattr(
            "app.core.prompt_provider_factory.os.path.exists",
            lambda _: True,
        )
        # Patch importlib.import_module LAST: it mutates the shared importlib
        # module's attribute (the factory calls importlib.import_module), which
        # under pytest>=9.1 would otherwise break monkeypatch.setattr's own
        # string-target resolution for any setattr that follows it.
        monkeypatch.setattr(
            "app.core.prompt_provider_factory.importlib.import_module",
            fake_import_module,
        )

        def register(root: str, module_name: str, provider_class: type) -> None:
            yielded_modules[root].append(module_name)
            mod = types.ModuleType(module_name)
            setattr(mod, provider_class.__name__, provider_class)
            fake_modules[module_name] = mod

        return register

    @staticmethod
    def _make_stub_provider(stub_name: str, caps: dict[str, str] | None = None) -> type:
        """Construct a minimal IJarvisPromptProvider subclass for testing."""
        capabilities = caps or {
            "model_family": "stub",
            "size_tier": "small",
            "training_tier": "untrained",
        }

        class _StubProvider(IJarvisPromptProvider):
            @property
            def name(self) -> str:
                return stub_name

            def build_system_prompt(
                self, node_context, timezone, tools, available_commands=None
            ) -> str:
                return ""

            def get_capabilities(self):
                return {
                    "provider_name": stub_name,
                    "use_tool_classifier": True,
                    **capabilities,
                }

        _StubProvider.__name__ = stub_name
        return _StubProvider

    # ── Regression: built-in root still works after the refactor ─────

    def test_builtin_root_provider_still_discovered_by_create_provider(self):
        """Real built-in provider must remain discoverable via create_provider."""
        provider = PromptProviderFactory.create_provider("LlamaSmallUntrained")
        assert provider is not None
        assert provider.name == "LlamaSmallUntrained"
        assert isinstance(provider, IJarvisPromptProvider)

    def test_builtin_root_provider_still_in_available_providers(self):
        """Real built-in provider must remain in get_available_providers()."""
        providers = PromptProviderFactory.get_available_providers()
        assert "LlamaSmallUntrained" in providers
        assert providers == sorted(providers)

    # ── Custom root discovery via mocked scan ────────────────────────

    def test_create_provider_finds_provider_in_custom_root(self, mock_scan):
        """Provider registered in custom root is returned by create_provider."""
        stub_cls = self._make_stub_provider("StubCustomProvider")
        mock_scan(
            "prompt_providers_custom",
            "app.core.prompt_providers_custom.small.untrained.stubpkg.provider",
            stub_cls,
        )

        provider = PromptProviderFactory.create_provider("StubCustomProvider")
        assert provider is not None
        assert provider.name == "StubCustomProvider"
        assert isinstance(provider, IJarvisPromptProvider)

    def test_create_provider_custom_root_case_insensitive(self, mock_scan):
        """Custom-root match honors case-insensitive contract."""
        stub_cls = self._make_stub_provider("StubCustomProvider")
        mock_scan(
            "prompt_providers_custom",
            "app.core.prompt_providers_custom.small.untrained.stubpkg.provider",
            stub_cls,
        )

        for query in ("stubcustomprovider", "STUBCUSTOMPROVIDER", "StubCustomProvider"):
            provider = PromptProviderFactory.create_provider(query)
            assert provider is not None, f"query {query!r} returned None"
            assert provider.name == "StubCustomProvider"

    def test_get_available_providers_includes_custom_root_names(self, mock_scan):
        """get_available_providers walks both roots and returns names from each."""
        builtin = self._make_stub_provider("BuiltinStub")
        custom = self._make_stub_provider("CustomStub")
        mock_scan(
            "prompt_providers",
            "app.core.prompt_providers.small.untrained.builtin.provider",
            builtin,
        )
        mock_scan(
            "prompt_providers_custom",
            "app.core.prompt_providers_custom.small.untrained.custom.provider",
            custom,
        )

        names = PromptProviderFactory.get_available_providers()
        assert "BuiltinStub" in names
        assert "CustomStub" in names
        assert names == sorted(names)

    def test_get_provider_info_works_for_custom_root_provider(self, mock_scan):
        """Admin/health endpoints can introspect a custom-root provider."""
        stub_cls = self._make_stub_provider(
            "CustomStub",
            caps={"model_family": "stub", "size_tier": "small", "training_tier": "untrained"},
        )
        mock_scan(
            "prompt_providers_custom",
            "app.core.prompt_providers_custom.small.untrained.stubpkg.provider",
            stub_cls,
        )

        info = PromptProviderFactory.get_provider_info("CustomStub")
        assert info is not None
        assert info["name"] == "CustomStub"
        assert info["capabilities"]["size_tier"] == "small"
        assert info["capabilities"]["model_family"] == "stub"

    # ── Deduplication and first-match behavior ───────────────────────

    def test_get_available_providers_dedupes_when_same_name_in_both_roots(self, mock_scan):
        """A name present in both roots must appear exactly once in the list."""
        builtin_dup = self._make_stub_provider("DuplicateStub")
        custom_dup = self._make_stub_provider("DuplicateStub")
        mock_scan(
            "prompt_providers",
            "app.core.prompt_providers.small.untrained.dup.provider",
            builtin_dup,
        )
        mock_scan(
            "prompt_providers_custom",
            "app.core.prompt_providers_custom.small.untrained.dup.provider",
            custom_dup,
        )

        names = PromptProviderFactory.get_available_providers()
        assert names.count("DuplicateStub") == 1

    def test_create_provider_prefers_first_match_builtin_wins(self, mock_scan):
        """When same name exists in both roots, built-in is scanned first and wins."""
        builtin_dup = self._make_stub_provider("DuplicateStub")
        custom_dup = self._make_stub_provider("DuplicateStub")
        mock_scan(
            "prompt_providers",
            "app.core.prompt_providers.small.untrained.dup.provider",
            builtin_dup,
        )
        mock_scan(
            "prompt_providers_custom",
            "app.core.prompt_providers_custom.small.untrained.dup.provider",
            custom_dup,
        )

        provider = PromptProviderFactory.create_provider("DuplicateStub")
        # Both stubs are defined in tests/ so __module__ comparison won't
        # distinguish them — compare the underlying class instead.
        assert provider is not None
        assert isinstance(provider, builtin_dup)
        assert not isinstance(provider, custom_dup)

    # ── Missing / empty custom root must not crash ───────────────────

    def test_create_provider_returns_none_when_custom_dir_missing(self, monkeypatch, tmp_path):
        """Factory must not crash when prompt_providers_custom/ is absent on disk."""
        # Point factory's dirname() at a tmp dir containing only prompt_providers/
        monkeypatch.setattr(
            "app.core.prompt_provider_factory.os.path.dirname",
            lambda _: str(tmp_path),
        )
        (tmp_path / "prompt_providers").mkdir()
        # prompt_providers_custom/ deliberately not created

        provider = PromptProviderFactory.create_provider("AnythingNotThere")
        assert provider is None

        names = PromptProviderFactory.get_available_providers()
        assert names == []

    def test_get_available_providers_empty_when_both_dirs_empty(self, monkeypatch, tmp_path):
        """Both roots present but empty → empty available-providers list."""
        monkeypatch.setattr(
            "app.core.prompt_provider_factory.os.path.dirname",
            lambda _: str(tmp_path),
        )
        (tmp_path / "prompt_providers").mkdir()
        (tmp_path / "prompt_providers_custom").mkdir()

        assert PromptProviderFactory.get_available_providers() == []
