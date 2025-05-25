from enum import Enum


class DatasetType(str, Enum):
    BIAS = "bias"
    UNBIAS = "unbias"


class TextFormat(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    FREE_TEXT = "free_text"
