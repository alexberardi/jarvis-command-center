"""Tests for prompt_provider_installer (#8).

Covers the sibling-dir refactor: installs land at
``prompt_providers_custom/{size}/{training}/{pkg}/`` (no ``custom/`` segment).
Filesystem-based tests against a tmp_path-rooted ``PROVIDERS_BASE``.

Network is mocked — ``subprocess.run`` (git clone) is replaced with a
``shutil.copytree`` from a prepared on-disk stub repo.
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from app.services import prompt_provider_installer
from app.services.prompt_provider_installer import (
    PromptProviderInstallError,
    install_prompt_provider,
    list_custom_providers,
    uninstall_prompt_provider,
)


# ── Stub provider source strings ─────────────────────────────────────

_STUB_PROVIDER_SOURCE = textwrap.dedent(
    """\
    from app.core.interfaces.ijarvis_prompt_provider import IJarvisPromptProvider


    class StubProvider(IJarvisPromptProvider):
        @property
        def name(self):
            return "StubProvider"

        def build_system_prompt(self, node_context, timezone, tools, available_commands=None):
            return ""

        def get_capabilities(self):
            return {
                "provider_name": "StubProvider",
                "model_family": "stub",
                "size_tier": "small",
                "training_tier": "untrained",
                "use_tool_classifier": True,
            }
    """
)


def _stub_provider_source_with_caps(size_tier: str, training_tier: str) -> str:
    """Build a stub provider whose get_capabilities() reports the given tiers."""
    return textwrap.dedent(
        f"""\
        from app.core.interfaces.ijarvis_prompt_provider import IJarvisPromptProvider


        class StubProvider(IJarvisPromptProvider):
            @property
            def name(self):
                return "StubProvider"

            def build_system_prompt(self, node_context, timezone, tools, available_commands=None):
                return ""

            def get_capabilities(self):
                return {{
                    "provider_name": "StubProvider",
                    "model_family": "stub",
                    "size_tier": {size_tier!r},
                    "training_tier": {training_tier!r},
                    "use_tool_classifier": True,
                }}
        """
    )


_MANIFEST = "name: stubpkg\nversion: 0.0.1\n"


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def patched_providers_base(monkeypatch, tmp_path):
    """Repoint PROVIDERS_BASE at a tmp custom root."""
    custom_root = tmp_path / "prompt_providers_custom"
    custom_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(prompt_provider_installer, "PROVIDERS_BASE", custom_root)
    return custom_root


@pytest.fixture
def stub_provider_source_tree(tmp_path):
    """Build a minimal valid stub provider repo on disk under tmp_path / "src"."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "jarvis_package.yaml").write_text(_MANIFEST, encoding="utf-8")
    (src / "provider.py").write_text(_STUB_PROVIDER_SOURCE, encoding="utf-8")
    return src


@pytest.fixture
def fake_git_clone(monkeypatch, stub_provider_source_tree):
    """Replace subprocess.run with a stub that copies the source tree to the clone target."""
    captured = {}

    def _fake_run(cmd, capture_output=False, text=False, timeout=None, **_kwargs):
        captured["cmd"] = cmd
        # The clone target is the last positional argument before any flags.
        # The installer always passes target last.
        target = Path(cmd[-1])
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(stub_provider_source_tree, target)

        result = subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return result

    monkeypatch.setattr(prompt_provider_installer.subprocess, "run", _fake_run)
    return captured


@pytest.fixture
def fake_git_clone_failure(monkeypatch):
    """subprocess.run that always reports git clone failure."""
    def _fake_run(cmd, capture_output=False, text=False, timeout=None, **_kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=128, stdout="", stderr="repository not found")

    monkeypatch.setattr(prompt_provider_installer.subprocess, "run", _fake_run)


@pytest.fixture
def factory_finds_installed(monkeypatch):
    """Short-circuit the post-install factory verification.

    The installer's post-install check at lines 124-137 calls
    ``PromptProviderFactory.create_provider(provider_name)`` which goes
    through the real factory. For unit tests we don't want to depend on
    Python's import machinery picking up the freshly written file —
    we stub the factory to return a sentinel instance for the expected name.
    """
    from app.core.interfaces.ijarvis_prompt_provider import IJarvisPromptProvider

    class _Sentinel(IJarvisPromptProvider):
        @property
        def name(self):
            return "StubProvider"

        def build_system_prompt(self, node_context, timezone, tools, available_commands=None):
            return ""

        def get_capabilities(self):
            return {"provider_name": "StubProvider"}

    def _fake_create_provider(cls, name=None):
        if name == "StubProvider":
            return _Sentinel()
        return None

    from app.core import prompt_provider_factory as ppf
    monkeypatch.setattr(
        ppf.PromptProviderFactory, "create_provider", classmethod(_fake_create_provider)
    )


# ── Happy path ──────────────────────────────────────────────────────


class TestInstallHappyPath:
    def test_install_lands_in_flat_layout_without_custom_segment(
        self, patched_providers_base, fake_git_clone, factory_finds_installed
    ):
        result = install_prompt_provider("https://example.invalid/stub.git")

        assert result["provider_name"] == "StubProvider"
        assert result["package_name"] == "stubpkg"
        assert result["size_tier"] == "small"
        assert result["training_tier"] == "untrained"
        # Flat layout: no "custom" segment in the relative install_path
        assert result["install_path"] == "small/untrained/stubpkg"
        assert "custom" not in result["install_path"]

        installed = patched_providers_base / "small" / "untrained" / "stubpkg"
        assert installed.is_dir()
        assert (installed / "provider.py").exists()
        assert (installed / "__init__.py").exists()
        # No "custom" intermediate directory was created
        assert not (patched_providers_base / "small" / "untrained" / "custom").exists()

    def test_install_invokes_git_clone(
        self, patched_providers_base, fake_git_clone, factory_finds_installed
    ):
        install_prompt_provider("https://example.invalid/stub.git", git_tag="v1.2.3")
        assert "cmd" in fake_git_clone
        assert "git" in fake_git_clone["cmd"][0]
        # --branch v1.2.3 should be present when git_tag is supplied
        assert "--branch" in fake_git_clone["cmd"]
        assert "v1.2.3" in fake_git_clone["cmd"]


# ── Idempotency / already-installed ────────────────────────────────


class TestInstallIdempotency:
    def test_install_raises_when_already_installed(
        self, patched_providers_base, fake_git_clone, factory_finds_installed
    ):
        pre_existing = patched_providers_base / "small" / "untrained" / "stubpkg"
        pre_existing.mkdir(parents=True)
        sentinel = pre_existing / "marker.txt"
        sentinel.write_text("preserve-me", encoding="utf-8")

        with pytest.raises(PromptProviderInstallError, match="already installed"):
            install_prompt_provider("https://example.invalid/stub.git")

        # Rollback must NOT remove a directory it didn't create
        assert pre_existing.is_dir()
        assert sentinel.read_text(encoding="utf-8") == "preserve-me"


# ── Validation errors ─────────────────────────────────────────────


class TestInstallValidationErrors:
    def test_install_raises_on_invalid_size_tier(
        self, patched_providers_base, monkeypatch, tmp_path
    ):
        # Build a stub that reports an unrecognized size tier
        src = tmp_path / "src_bad_size"
        src.mkdir()
        (src / "jarvis_package.yaml").write_text(_MANIFEST, encoding="utf-8")
        (src / "provider.py").write_text(
            _stub_provider_source_with_caps("xl", "untrained"), encoding="utf-8"
        )

        def _fake_run(cmd, capture_output=False, text=False, timeout=None, **_kwargs):
            target = Path(cmd[-1])
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(prompt_provider_installer.subprocess, "run", _fake_run)

        with pytest.raises(PromptProviderInstallError, match="Invalid size_tier"):
            install_prompt_provider("https://example.invalid/bad.git")

    def test_install_raises_on_invalid_training_tier(
        self, patched_providers_base, monkeypatch, tmp_path
    ):
        src = tmp_path / "src_bad_training"
        src.mkdir()
        (src / "jarvis_package.yaml").write_text(_MANIFEST, encoding="utf-8")
        (src / "provider.py").write_text(
            _stub_provider_source_with_caps("small", "wonky"), encoding="utf-8"
        )

        def _fake_run(cmd, capture_output=False, text=False, timeout=None, **_kwargs):
            target = Path(cmd[-1])
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(prompt_provider_installer.subprocess, "run", _fake_run)

        with pytest.raises(PromptProviderInstallError, match="Invalid training_tier"):
            install_prompt_provider("https://example.invalid/bad.git")

    def test_install_raises_on_git_clone_failure(
        self, patched_providers_base, fake_git_clone_failure
    ):
        with pytest.raises(PromptProviderInstallError, match="Git clone failed"):
            install_prompt_provider("https://example.invalid/nope.git")

        # Nothing should have been created under PROVIDERS_BASE
        assert list(patched_providers_base.iterdir()) == []

    def test_install_raises_when_no_manifest_found(
        self, patched_providers_base, monkeypatch, tmp_path
    ):
        src = tmp_path / "src_no_manifest"
        src.mkdir()
        # No jarvis_package.yaml / jarvis_command.yaml
        (src / "provider.py").write_text(_STUB_PROVIDER_SOURCE, encoding="utf-8")

        def _fake_run(cmd, capture_output=False, text=False, timeout=None, **_kwargs):
            target = Path(cmd[-1])
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(prompt_provider_installer.subprocess, "run", _fake_run)

        with pytest.raises(PromptProviderInstallError, match="No manifest"):
            install_prompt_provider("https://example.invalid/x.git")

    def test_install_rollback_on_factory_verification_failure(
        self, patched_providers_base, fake_git_clone, monkeypatch
    ):
        # Make the factory return None even though install succeeds
        from app.core import prompt_provider_factory as ppf

        def _fake_create_provider(cls, name=None):
            return None

        monkeypatch.setattr(
            ppf.PromptProviderFactory,
            "create_provider",
            classmethod(_fake_create_provider),
        )

        with pytest.raises(PromptProviderInstallError, match="factory cannot discover"):
            install_prompt_provider("https://example.invalid/stub.git")

        # Rollback removed the partially-installed package dir
        installed = patched_providers_base / "small" / "untrained" / "stubpkg"
        assert not installed.exists()


# ── list_custom_providers ────────────────────────────────────────


class TestListCustomProviders:
    def test_list_empty_when_no_packages(self, patched_providers_base):
        assert list_custom_providers() == []

    def test_list_reports_flat_layout_paths(self, patched_providers_base):
        # Drop two stub packages directly under the flat layout
        pkg1 = patched_providers_base / "small" / "untrained" / "alpha"
        pkg2 = patched_providers_base / "medium" / "trained" / "beta"
        pkg1.mkdir(parents=True)
        pkg2.mkdir(parents=True)
        (pkg1 / "provider.py").write_text("# alpha", encoding="utf-8")
        (pkg2 / "provider.py").write_text("# beta", encoding="utf-8")

        results = list_custom_providers()
        assert len(results) == 2

        by_name = {r["package_name"]: r for r in results}
        assert by_name["alpha"]["size_tier"] == "small"
        assert by_name["alpha"]["training_tier"] == "untrained"
        assert by_name["alpha"]["path"] == "small/untrained/alpha"
        assert "custom" not in by_name["alpha"]["path"]

        assert by_name["beta"]["size_tier"] == "medium"
        assert by_name["beta"]["training_tier"] == "trained"
        assert by_name["beta"]["path"] == "medium/trained/beta"
        assert "custom" not in by_name["beta"]["path"]

    def test_list_skips_dotfiles_and_underscore_dirs(self, patched_providers_base):
        st_dir = patched_providers_base / "small" / "untrained"
        st_dir.mkdir(parents=True)
        # Skipped entries
        (st_dir / ".gitkeep").write_text("", encoding="utf-8")
        (st_dir / "__pycache__").mkdir()
        (st_dir / "_helper").mkdir()
        # Real package
        real = st_dir / "realpkg"
        real.mkdir()
        (real / "provider.py").write_text("# real", encoding="utf-8")

        results = list_custom_providers()
        assert len(results) == 1
        assert results[0]["package_name"] == "realpkg"

    def test_list_skips_packages_without_provider_py(self, patched_providers_base):
        st_dir = patched_providers_base / "small" / "untrained"
        st_dir.mkdir(parents=True)
        # No provider.py — should be ignored, not crash
        (st_dir / "halfbaked").mkdir()

        assert list_custom_providers() == []


# ── uninstall_prompt_provider ────────────────────────────────────


class TestUninstallPromptProvider:
    def test_uninstall_returns_false_for_unknown_package(self, patched_providers_base):
        assert uninstall_prompt_provider("ghost") is False

    def test_uninstall_removes_package_under_flat_layout(self, patched_providers_base):
        pkg = patched_providers_base / "small" / "untrained" / "stubpkg"
        pkg.mkdir(parents=True)
        (pkg / "provider.py").write_text("# stub", encoding="utf-8")

        assert uninstall_prompt_provider("stubpkg") is True
        assert not pkg.exists()
        # Parent size/training dirs stay (we only remove the package leaf)
        assert pkg.parent.exists()

    def test_uninstall_searches_all_size_training_combos(self, patched_providers_base):
        pkg = patched_providers_base / "medium" / "trained" / "stubpkg"
        pkg.mkdir(parents=True)
        (pkg / "provider.py").write_text("# stub", encoding="utf-8")

        assert uninstall_prompt_provider("stubpkg") is True
        assert not pkg.exists()
