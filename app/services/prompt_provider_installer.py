"""Prompt provider installer — clone from GitHub, validate, install to CC.

Community prompt providers are installed into
app/core/prompt_providers/{size_tier}/{training_tier}/custom/{name}/
and discovered at runtime by PromptProviderFactory via pkgutil.walk_packages.
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from app.core.interfaces.ijarvis_prompt_provider import IJarvisPromptProvider

logger = logging.getLogger("uvicorn")

PROVIDERS_BASE = Path(__file__).resolve().parents[1] / "core" / "prompt_providers"

VALID_SIZE_TIERS = {"small", "medium", "large"}
VALID_TRAINING_TIERS = {"untrained", "trained"}


class PromptProviderInstallError(Exception):
    pass


def install_prompt_provider(github_repo_url: str, git_tag: str | None = None) -> dict[str, Any]:
    """Clone a prompt provider repo, validate it, and install to CC.

    Args:
        github_repo_url: GitHub clone URL.
        git_tag: Optional git tag to checkout.

    Returns:
        Dict with install result metadata.

    Raises:
        PromptProviderInstallError: On any failure (with cleanup).
    """
    tmpdir = tempfile.mkdtemp(prefix="jarvis-pp-")
    install_path: Path | None = None

    try:
        # Clone
        clone_cmd = ["git", "clone", "--depth", "1"]
        if git_tag:
            clone_cmd += ["--branch", git_tag]
        clone_cmd += [github_repo_url, tmpdir]

        result = subprocess.run(
            clone_cmd, capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            raise PromptProviderInstallError(f"Git clone failed: {result.stderr.strip()}")

        repo_dir = Path(tmpdir)

        # Find manifest
        import yaml
        manifest: dict[str, Any] | None = None
        for manifest_name in ("jarvis_package.yaml", "jarvis_command.yaml"):
            manifest_path = repo_dir / manifest_name
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = yaml.safe_load(f)
                break

        if not manifest:
            raise PromptProviderInstallError("No manifest file found")

        # Find provider.py (root or prompt_providers/{name}/)
        provider_file = _find_provider_file(repo_dir, manifest)
        if not provider_file:
            raise PromptProviderInstallError(
                "No provider.py found. Expected at root or in prompt_providers/{name}/"
            )

        # Load and validate the provider class
        provider_info = _validate_provider(provider_file)
        capabilities = provider_info["capabilities"]

        size_tier = capabilities.get("size_tier", "medium")
        training_tier = capabilities.get("training_tier", "untrained")

        if size_tier not in VALID_SIZE_TIERS:
            raise PromptProviderInstallError(
                f"Invalid size_tier '{size_tier}' — must be one of {VALID_SIZE_TIERS}"
            )
        if training_tier not in VALID_TRAINING_TIERS:
            raise PromptProviderInstallError(
                f"Invalid training_tier '{training_tier}' — must be one of {VALID_TRAINING_TIERS}"
            )

        provider_name = provider_info["name"]
        package_name = manifest.get("name", provider_name.lower())

        # Install destination
        install_dir = PROVIDERS_BASE / size_tier / training_tier / "custom" / package_name
        if install_dir.exists():
            raise PromptProviderInstallError(
                f"Provider already installed at {install_dir.relative_to(PROVIDERS_BASE)}"
            )

        # Copy provider files
        install_dir.mkdir(parents=True, exist_ok=True)
        install_path = install_dir

        # Copy the provider file
        shutil.copy2(provider_file, install_dir / "provider.py")

        # Copy any sibling files (shared helpers in the same directory)
        provider_parent = provider_file.parent
        for f in provider_parent.iterdir():
            if f.is_file() and f.suffix == ".py" and f.name != provider_file.name:
                shutil.copy2(f, install_dir / f.name)

        # Ensure __init__.py exists
        init_file = install_dir / "__init__.py"
        if not init_file.exists():
            init_file.touch()

        # Verify the installed provider can be discovered by the factory
        try:
            from app.core.prompt_provider_factory import PromptProviderFactory
            found = PromptProviderFactory.create_provider(provider_name)
            if found is None:
                raise PromptProviderInstallError(
                    f"Provider installed but factory cannot discover '{provider_name}'"
                )
        except PromptProviderInstallError:
            raise
        except Exception as e:
            raise PromptProviderInstallError(
                f"Provider installed but factory validation failed: {e}"
            )

        logger.info(
            "Installed prompt provider: %s → %s/%s/custom/%s",
            provider_name, size_tier, training_tier, package_name,
        )

        return {
            "provider_name": provider_name,
            "package_name": package_name,
            "size_tier": size_tier,
            "training_tier": training_tier,
            "install_path": str(install_dir.relative_to(PROVIDERS_BASE)),
            "capabilities": capabilities,
        }

    except PromptProviderInstallError:
        # Rollback on failure
        if install_path and install_path.exists():
            shutil.rmtree(install_path, ignore_errors=True)
        raise
    except Exception as e:
        if install_path and install_path.exists():
            shutil.rmtree(install_path, ignore_errors=True)
        raise PromptProviderInstallError(f"Unexpected error: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def list_custom_providers() -> list[dict[str, Any]]:
    """List all installed custom prompt providers."""
    providers: list[dict[str, Any]] = []

    for size_tier in VALID_SIZE_TIERS:
        for training_tier in VALID_TRAINING_TIERS:
            custom_dir = PROVIDERS_BASE / size_tier / training_tier / "custom"
            if not custom_dir.is_dir():
                continue
            for pkg_dir in sorted(custom_dir.iterdir()):
                if not pkg_dir.is_dir() or pkg_dir.name.startswith(("_", ".")):
                    continue
                provider_file = pkg_dir / "provider.py"
                if provider_file.exists():
                    providers.append({
                        "package_name": pkg_dir.name,
                        "size_tier": size_tier,
                        "training_tier": training_tier,
                        "path": str(pkg_dir.relative_to(PROVIDERS_BASE)),
                    })

    return providers


def uninstall_prompt_provider(package_name: str) -> bool:
    """Remove a custom prompt provider by package name.

    Returns True if removed, False if not found.
    """
    for size_tier in VALID_SIZE_TIERS:
        for training_tier in VALID_TRAINING_TIERS:
            install_dir = PROVIDERS_BASE / size_tier / training_tier / "custom" / package_name
            if install_dir.exists():
                shutil.rmtree(install_dir)
                logger.info("Uninstalled prompt provider: %s from %s/%s", package_name, size_tier, training_tier)
                return True

    return False


def _find_provider_file(repo_dir: Path, manifest: dict[str, Any]) -> Path | None:
    """Find the provider.py entry point in the repo."""
    # Check explicit components in manifest
    components = manifest.get("components", [])
    for comp in components:
        if comp.get("type") == "prompt_provider":
            path = repo_dir / comp.get("path", "")
            if path.exists():
                return path

    # Convention: prompt_providers/{name}/provider.py
    pp_dir = repo_dir / "prompt_providers"
    if pp_dir.is_dir():
        for sub_dir in pp_dir.iterdir():
            if sub_dir.is_dir() and not sub_dir.name.startswith(("_", ".")):
                entry = sub_dir / "provider.py"
                if entry.exists():
                    return entry

    # Root-level provider.py
    root_provider = repo_dir / "provider.py"
    if root_provider.exists():
        return root_provider

    return None


def _validate_provider(provider_file: Path) -> dict[str, Any]:
    """Load and validate a provider file, return its metadata.

    We exec the file in an isolated namespace and check that it defines
    a class with the required methods.
    """
    import ast

    source = provider_file.read_text(encoding="utf-8")

    # AST check first
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise PromptProviderInstallError(f"Syntax error in provider.py: {e}")

    # Find class inheriting from IJarvisPromptProvider
    found_class = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = None
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name and "PromptProvider" in base_name:
                    found_class = True
                    # Check required methods
                    defined = {
                        n.name for n in node.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    }
                    required = {"name", "build_system_prompt", "get_capabilities"}
                    missing = required - defined
                    if missing:
                        raise PromptProviderInstallError(
                            f"Provider missing required methods: {', '.join(sorted(missing))}"
                        )
                    break

    if not found_class:
        raise PromptProviderInstallError(
            "No class inheriting from IJarvisPromptProvider found in provider.py"
        )

    # Try to instantiate (to get capabilities and name)
    # We do this by exec-ing in a controlled namespace
    try:
        namespace: dict[str, Any] = {}
        # Provide the base class in the namespace
        namespace["IJarvisPromptProvider"] = IJarvisPromptProvider
        exec(compile(source, str(provider_file), "exec"), namespace)

        # Find the provider class
        provider_cls = None
        for obj in namespace.values():
            if (
                isinstance(obj, type)
                and issubclass(obj, IJarvisPromptProvider)
                and obj is not IJarvisPromptProvider
            ):
                provider_cls = obj
                break

        if provider_cls is None:
            raise PromptProviderInstallError("Could not find provider class after exec")

        instance = provider_cls()
        return {
            "name": instance.name,
            "capabilities": instance.get_capabilities(),
        }
    except PromptProviderInstallError:
        raise
    except Exception as e:
        raise PromptProviderInstallError(f"Provider instantiation failed: {e}")
