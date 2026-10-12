"""
Tool router classifier interface and fastText implementation.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("uvicorn")


@dataclass(frozen=True)
class ToolPrediction:
    tool_name: Optional[str]
    score: float


class ToolClassifier:
    """Abstract classifier interface."""

    def predict(self, utterance: str) -> ToolPrediction:
        raise NotImplementedError


class FastTextToolClassifier(ToolClassifier):
    """fastText-based classifier for tool routing."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        try:
            import fasttext  # type: ignore
        except Exception as exc:
            raise RuntimeError("fasttext is not installed") from exc
        self._model = fasttext.load_model(self.model_path)

    def _predict(self, utterance: str) -> tuple[tuple[str, ...], tuple[float, ...]]:
        """Top-1 (labels, scores) from the loaded model.

        fastText 0.9.x's Python ``predict()`` calls ``np.array(probs, copy=False)``,
        which raises ``ValueError`` under NumPy 2, so every prediction failed
        (and was swallowed as "classifier unavailable"). The C++ binding it wraps
        returns ``(score, label)`` pairs directly and has no such problem, so fall
        back to it. fastText also rejects input containing a newline.
        """
        text = utterance.replace("\n", " ")
        try:
            return self._model.predict(text, k=1)  # type: ignore[union-attr]
        except ValueError as exc:
            if "copy" not in str(exc).lower():
                raise
        pairs = self._model.f.predict(text + "\n", 1, 0.0, "strict")  # type: ignore[union-attr]
        return tuple(label for _score, label in pairs), tuple(score for score, _label in pairs)

    def predict(self, utterance: str) -> ToolPrediction:
        if not utterance:
            return ToolPrediction(tool_name=None, score=0.0)
        if not self.model_path:
            return ToolPrediction(tool_name=None, score=0.0)
        try:
            self._load()
            labels, scores = self._predict(utterance)
        except Exception as exc:
            logger.warning("⚠️ Tool classifier unavailable: %s", exc)
            return ToolPrediction(tool_name=None, score=0.0)
        if not labels or not scores:
            return ToolPrediction(tool_name=None, score=0.0)
        label = labels[0]
        score = float(scores[0])
        tool_name = label.replace("__label__", "")
        return ToolPrediction(tool_name=tool_name, score=score)

    @classmethod
    def from_env(cls) -> Optional["FastTextToolClassifier"]:
        model_path = os.getenv("JARVIS_TOOL_CLASSIFIER_MODEL_PATH", "").strip()
        if not model_path:
            return None
        return cls(model_path=model_path)
