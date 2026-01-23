from pydantic import BaseModel
from typing import List, Optional

from app.request_models.voice_command_request import CommandDefinition


class AdapterTrainingParams(BaseModel):
    rank: Optional[int] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    max_seq_len: Optional[int] = None
    hf_base_model_id: Optional[str] = None


class AdapterTrainingRequest(BaseModel):
    base_model_id: str
    available_commands: List[CommandDefinition]
    dataset_hash: Optional[str] = None
    params: Optional[AdapterTrainingParams] = None
    priority: Optional[str] = "normal"
