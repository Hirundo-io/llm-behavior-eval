from enum import Enum

from pydantic import BaseModel


class JudgeType(str, Enum):
    BIAS = "bias"


class EvaluationConfig(BaseModel):
    max_samples: None | int
    batch_size: int = 64
    judge_batch_size: int = 16
    judge_output_tokens: int = 32
    answer_tokens: int = 128
    num_return_sequences: int = 1
    sample: bool = False
    temperature: float = 0.6
    judge_type: JudgeType = JudgeType.BIAS
    model_path_or_repo_id: str
    judge_path_or_repo_id: str
    results_dir:str
