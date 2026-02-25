from enum import Enum


class DatasetType(str, Enum):
    BIAS = "bias"
    UNBIAS = "unbias"


class AnswerFormat(str, Enum):
    FREE_TEXT = "free_text"
    MULTIPLE_CHOICE = "multiple_choice"


# Supported bias types per source
# BBQ supports the following bias types
BBQ_BIAS_TYPES: set[str] = {
    "gender",
    "race",
    "nationality",
    "physical",
    "age",
    "religion",
}

# CBBQ supports the following bias types
CBBQ_BIAS_TYPES: set[str] = {
    "SES",
    "age",
    "disability",
    "disease",
    "educational_qualification",
    "ethnicity",
    "gender",
    "household_registration",
    "nationality",
    "physical_appearance",
    "race",
    "region",
    "religion",
    "sexual_orientation",
}

CBBQ_BASIC_TYPES: tuple[str, ...] = (
    "gender",
    "age",
    "disability",
    "SES",
)

CBBQ_ALL_TYPES: tuple[str, ...] = tuple(sorted(CBBQ_BIAS_TYPES, key=str.lower))

CBBQ_DIMENSION_LABELS: dict[str, str] = {
    "SES": "Socio-Economic Status (SES)",
    "age": "Age",
    "disability": "Disability",
    "disease": "Disease",
    "educational_qualification": "Educational Qualification",
    "ethnicity": "Ethnicity",
    "gender": "Gender",
    "household_registration": "Household Registration",
    "nationality": "Nationality",
    "physical_appearance": "Physical Appearance",
    "race": "Race",
    "region": "Region",
    "religion": "Religion",
    "sexual_orientation": "Sexual Orientation",
}

# UNQOVER supports the following bias types
UNQOVER_BIAS_TYPES: set[str] = {
    "religion",
    "gender",
    "race",
    "nationality",
}
