"""Answer Question Tool — server-side general knowledge answers.

Replaces the client-side ``answer_question`` command.  When the LLM calls
this tool, the engine formats the result via a clean LLM call (for
text-based models) or lets the model respond naturally (native tool mode).
"""

import logging
from typing import Any, Dict, Optional

from app.core.interfaces.iserver_tool import IServerTool

logger = logging.getLogger("uvicorn")


class AnswerQuestionTool(IServerTool):
    """Server-side tool for answering general knowledge questions."""

    @property
    def name(self) -> str:
        return "answer_question"

    @property
    def description(self) -> str:
        return (
            "Answer a general knowledge question about stable facts, definitions, "
            "history, science, geography, or biographies. NOT for current events, "
            "live scores, or time-sensitive information."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The knowledge question to answer",
                },
            },
            "required": ["query"],
        }

    def execute(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        user_utterance: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Return the query for the engine to answer via a formatting call.

        The tool itself doesn't generate the answer — the conversation handler
        makes a clean LLM call (without tool definitions) so the model answers
        from its own knowledge without entering a tool-call loop.
        """
        logger.info("📚 answer_question called: %s", query[:80])
        return {"query": query}
