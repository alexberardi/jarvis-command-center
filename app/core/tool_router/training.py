"""
Training utilities for the tool router classifier.
"""

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger("uvicorn")


@dataclass(frozen=True)
class TrainingExample:
    utterance: str
    tool_name: str


COMMAND_TEST_REGEX = re.compile(
    r"CommandTest\(\s*\"(?P<utterance>.*?)\"\s*,\s*\"(?P<tool>.*?)\"",
    re.DOTALL,
)


def _load_examples_from_test_results(path: Path) -> List[TrainingExample]:
    examples: List[TrainingExample] = []
    if not path.exists():
        return examples
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("⚠️ Failed to parse test results: %s", exc)
        return examples
    for item in payload.get("test_results", []):
        utterance = item.get("voice_command")
        tool_name = (item.get("expected") or {}).get("command")
        if utterance and tool_name:
            examples.append(TrainingExample(utterance=utterance, tool_name=tool_name))
    return examples


def _load_examples_from_test_script(path: Path) -> List[TrainingExample]:
    examples: List[TrainingExample] = []
    if not path.exists():
        return examples
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("⚠️ Failed to read test script: %s", exc)
        return examples
    for match in COMMAND_TEST_REGEX.finditer(text):
        utterance = match.group("utterance").strip()
        tool_name = match.group("tool").strip()
        if utterance and tool_name:
            examples.append(TrainingExample(utterance=utterance, tool_name=tool_name))
    return examples


def _parse_training_jsonl(jsonl_text: str) -> List[TrainingExample]:
    examples: List[TrainingExample] = []
    if not jsonl_text:
        return examples
    for line in jsonl_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        utterance = record.get("utterance")
        tool_name = record.get("tool_name")
        if utterance and tool_name:
            examples.append(TrainingExample(utterance=utterance, tool_name=tool_name))
    return examples


def _examples_from_commands(commands: Iterable[dict]) -> List[TrainingExample]:
    examples: List[TrainingExample] = []
    for cmd in commands:
        tool_name = cmd.get("command_name")
        for example in cmd.get("examples", []) or []:
            utterance = example.get("voice_command")
            if tool_name and utterance:
                examples.append(TrainingExample(utterance=utterance, tool_name=tool_name))
    return examples


def write_training_jsonl(examples: List[TrainingExample], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for ex in examples:
            handle.write(json.dumps({"utterance": ex.utterance, "tool_name": ex.tool_name}) + "\n")
    return path


def build_training_examples(
    repo_root: Path,
    extra_examples: Optional[Iterable[TrainingExample]] = None,
    extra_path: Optional[Path] = None,
    extra_jsonl: Optional[str] = None,
    command_payloads: Optional[Iterable[dict]] = None,
) -> List[TrainingExample]:
    examples: List[TrainingExample] = []
    examples.extend(_load_examples_from_test_results(repo_root / "temp" / "test_results.json"))
    examples.extend(_load_examples_from_test_script(repo_root / "temp" / "test_command_parsing.py"))
    if command_payloads:
        examples.extend(_examples_from_commands(command_payloads))
    if extra_path and extra_path.exists():
        try:
            for line in extra_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                utterance = record.get("utterance")
                tool_name = record.get("tool_name")
                if utterance and tool_name:
                    examples.append(TrainingExample(utterance=utterance, tool_name=tool_name))
        except Exception as exc:
            logger.warning("⚠️ Failed to load extra training data: %s", exc)
    if extra_jsonl:
        examples.extend(_parse_training_jsonl(extra_jsonl))
    if extra_examples:
        examples.extend(list(extra_examples))
    logger.info("📚 Loaded %d training examples", len(examples))
    return examples


def train_fasttext_classifier(
    examples: List[TrainingExample],
    output_path: Path,
    epoch: int = 25,
    lr: float = 0.5,
    word_ngrams: int = 2,
) -> Path:
    if not examples:
        raise ValueError("No training examples provided")
    try:
        import fasttext  # type: ignore
    except Exception as exc:
        raise RuntimeError("fasttext is not installed") from exc

    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as fp:
        for ex in examples:
            label = f"__label__{ex.tool_name}"
            fp.write(f"{label} {ex.utterance}\n")
        train_path = fp.name

    logger.info("🧪 Training fastText model on %d examples", len(examples))
    model = fasttext.train_supervised(
        input=train_path,
        epoch=epoch,
        lr=lr,
        wordNgrams=word_ngrams,
        verbose=2,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    logger.info("✅ Saved classifier model to %s", output_path)
    return output_path


def train_from_repo(
    repo_root: Path,
    output_path: Optional[Path] = None,
) -> Path:
    if output_path is None:
        output_path = repo_root / "temp" / "tool_classifier.bin"
    extra_path_env = os.getenv("JARVIS_TOOL_CLASSIFIER_EXTRA_TRAINING_PATH", "").strip()
    extra_path = Path(extra_path_env) if extra_path_env else None
    examples = build_training_examples(repo_root, extra_path=extra_path)
    return train_fasttext_classifier(examples, output_path)


if __name__ == "__main__":
    repo_root = Path(os.getenv("JARVIS_REPO_ROOT", Path(__file__).resolve().parents[3]))
    output = os.getenv("JARVIS_TOOL_CLASSIFIER_MODEL_PATH")
    output_path = Path(output) if output else None
    train_from_repo(repo_root, output_path=output_path)
