"""Tests for the Phase 4 training orchestrator."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from app.services.training_data_extractor import ExtractedRow, to_dataset_ref_row
from app.services.training_orchestrator import (
    build_dataset_ref,
    orchestrate_training,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _row(tool_name: str = "music", user_id: int = 1, msg: str = "play jazz",
         args: dict | None = None, source: str = "explicit_positive") -> ExtractedRow:
    return ExtractedRow(
        user_id=user_id,
        household_id="h1",
        conversation_id="c1",
        created_at=datetime(2026, 4, 19, 0, 0, 0),
        user_message=msg,
        tool_call={"name": tool_name, "arguments": args or {"action": "play"}},
        source=source,
        user_rating=None,
    )


def _format(row: ExtractedRow) -> dict:
    return to_dataset_ref_row(row, system_prompt="SYS")


# --------------------------------------------------------------------------
# build_dataset_ref
# --------------------------------------------------------------------------


class TestDatasetRef:
    def test_groups_by_command(self):
        rows = [
            _format(_row(tool_name="music")),
            _format(_row(tool_name="music")),
            _format(_row(tool_name="set_timer", args={"duration_minutes": 5})),
        ]
        ref = build_dataset_ref(rows)
        assert ref["format"] == "inline-json"
        cmds = {c["command_name"]: len(c["examples"]) for c in ref["data"]["commands"]}
        assert cmds == {"music": 2, "set_timer": 1}

    def test_empty(self):
        ref = build_dataset_ref([])
        assert ref["data"]["commands"] == []


# --------------------------------------------------------------------------
# Full orchestrator flow — enqueue + poll mocked
# --------------------------------------------------------------------------


class TestOrchestrateTraining:
    @pytest.fixture(autouse=True)
    def _mock_net(self):
        """Intercept httpx so no real network calls."""
        with patch("app.services.training_orchestrator.enqueue_training") as enq, \
             patch("app.services.training_orchestrator.poll_status") as poll:
            enq.return_value = "job-abc"
            poll.return_value = {
                "status": "COMPLETE",
                "dataset_hash": "hash-xyz",
                "started_at": "2026-04-19T00:00:00+00:00",
                "completed_at": "2026-04-19T00:05:00+00:00",
            }
            self.enq_mock = enq
            self.poll_mock = poll
            yield

    def test_happy_path_no_eval(self):
        result = orchestrate_training(
            organic_source=lambda: [_row() for _ in range(3)],
            synthetic_source=lambda: [_row(source="synthetic_expansion") for _ in range(2)],
            formatter=_format,
            llm_proxy_url="http://localhost:7704",
            base_model_id="test-model",
            node_id="node-1",
            app_id="x", app_key="y",
            training_params={"epochs": 1, "lora_r": 16},
            eval_runner=None,
        )
        assert result.status == "success"
        assert result.adapter_hash == "hash-xyz"
        assert result.job_id == "job-abc"
        assert result.training_seconds == 300.0  # 5 min
        assert result.dataset_rows == 5
        assert result.by_source == {"explicit_positive": 3, "synthetic_expansion": 2}
        assert result.eval is None
        self.enq_mock.assert_called_once()
        self.poll_mock.assert_called_once()

    def test_returns_enqueue_failed_when_no_rows(self):
        result = orchestrate_training(
            organic_source=lambda: [],
            synthetic_source=lambda: [],
            formatter=_format,
            llm_proxy_url="http://localhost:7704",
            base_model_id="m", node_id="n",
            app_id="x", app_key="y",
            training_params={},
            eval_runner=None,
        )
        assert result.status == "enqueue_failed"
        assert "no rows" in (result.error or "")
        self.enq_mock.assert_not_called()
        self.poll_mock.assert_not_called()

    def test_runs_eval_when_runner_provided(self):
        eval_runner = MagicMock(return_value={"verdict": "PASS", "pass_rate": 91.7})
        result = orchestrate_training(
            organic_source=lambda: [_row()],
            synthetic_source=None,
            formatter=_format,
            llm_proxy_url="http://localhost:7704",
            base_model_id="m", node_id="n",
            app_id="x", app_key="y",
            training_params={},
            eval_runner=eval_runner,
        )
        assert result.status == "success"
        assert result.eval == {"verdict": "PASS", "pass_rate": 91.7}
        eval_runner.assert_called_once_with("hash-xyz")

    def test_eval_failure_path(self):
        def boom(h):
            raise RuntimeError("eval blew up")
        result = orchestrate_training(
            organic_source=lambda: [_row()],
            synthetic_source=None,
            formatter=_format,
            llm_proxy_url="http://localhost:7704",
            base_model_id="m", node_id="n",
            app_id="x", app_key="y",
            training_params={},
            eval_runner=boom,
        )
        assert result.status == "eval_failed"
        assert "eval blew up" in (result.error or "")
        assert result.adapter_hash == "hash-xyz"  # training still succeeded

    def test_training_failed_propagates(self):
        self.poll_mock.return_value = {
            "status": "FAILED",
            "error_message": "CalledProcessError returncode=1",
        }
        result = orchestrate_training(
            organic_source=lambda: [_row()],
            synthetic_source=None,
            formatter=_format,
            llm_proxy_url="http://localhost:7704",
            base_model_id="m", node_id="n",
            app_id="x", app_key="y",
            training_params={},
            eval_runner=None,
        )
        assert result.status == "training_failed"
        assert "CalledProcessError" in (result.error or "")

    def test_enqueue_exception(self):
        self.enq_mock.side_effect = RuntimeError("502 bad gateway")
        result = orchestrate_training(
            organic_source=lambda: [_row()],
            synthetic_source=None,
            formatter=_format,
            llm_proxy_url="http://localhost:7704",
            base_model_id="m", node_id="n",
            app_id="x", app_key="y",
            training_params={},
            eval_runner=None,
        )
        assert result.status == "enqueue_failed"
        assert "502" in (result.error or "")
