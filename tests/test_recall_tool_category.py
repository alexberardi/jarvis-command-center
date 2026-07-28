"""The recall tool's "general" category means "no filter", not a literal.

2026-07-27 prod: "Do you know who Leo is?" → the model called
recall(query="Leo", category="general"). The Leo memory is categorized
"fact", and both search paths applied category as a literal equality
filter — so a memory whose content literally contains "Leo" scored zero
results on both the vector AND substring paths. The model reaches for
"general" whenever it has no category opinion (it's in the enum), so it
must behave as a wildcard.
"""

from unittest.mock import MagicMock, patch

from app.core.tools.recall_tool import RecallTool


def _run_recall(**tool_args):
    """Execute the tool with mocked context + DB, returning the category
    each search path actually received."""
    tool = RecallTool()
    seen: dict = {}

    def _vector_search(svc, user_id, household_id, query, limit, category, threshold):
        seen["vector_category"] = category
        return []  # empty (not None) → no substring fallback needed

    mock_cache = MagicMock()
    mock_cache.get_node_context.return_value = {
        "speaker_user_id": 1,
        "household_id": "hh-1",
    }

    with patch("app.core.tools.recall_tool.conversation_cache", mock_cache), \
         patch.object(RecallTool, "_get_recall_settings", return_value=(5, 0.3)), \
         patch.object(RecallTool, "_vector_search", side_effect=_vector_search), \
         patch("app.db.get_session_local") as mock_sl:
        mock_sl.return_value.return_value = MagicMock()
        result = tool.execute(conversation_id="conv-1", **tool_args)
    return seen, result


class TestGeneralCategoryIsWildcard:
    def test_general_is_not_passed_as_a_filter(self):
        seen, _ = _run_recall(query="Leo", category="general")
        assert seen["vector_category"] is None

    def test_real_categories_still_filter(self):
        seen, _ = _run_recall(query="coffee", category="preference")
        assert seen["vector_category"] == "preference"

    def test_no_category_stays_unfiltered(self):
        seen, _ = _run_recall(query="Leo")
        assert seen["vector_category"] is None
