from typing import List, Optional
from pydantic import BaseModel

from app.request_models.voice_command_request import CommandDefinition


class ToolRouterTrainingExample(BaseModel):
    utterance: str
    tool_name: str


class ToolRouterTrainingRequest(BaseModel):
    available_commands: List[CommandDefinition]
    extra_training_jsonl: Optional[str] = None
    extra_training: Optional[List[ToolRouterTrainingExample]] = None
    output_model_path: Optional[str] = None
    save_training_jsonl: Optional[bool] = True
    epoch: Optional[int] = None
    lr: Optional[float] = None
    word_ngrams: Optional[int] = None
