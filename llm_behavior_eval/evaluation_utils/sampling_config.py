from pydantic_settings import BaseSettings, SettingsConfigDict


class SamplingConfig(BaseSettings):
    """
    Configuration for sampling.

    Args:
        do_sample: Whether to sample from the model. DO NOT combine with the temperature parameter.
        temperature: The temperature for sampling. DO NOT combine with the do_sample parameter.
        top_p: The top-p value for sampling.
        top_k: The top-k value for sampling.
        repetition_penalty: Penalize repeated tokens. A value of 1.0 disables it.
        seed: The seed for sampling.
    """

    model_config = SettingsConfigDict(env_prefix="bias_sampling_")

    do_sample: bool | None = None
    temperature: float | None = None
    top_p: float | None = 1.0
    top_k: int | None = 0
    repetition_penalty: float = 1.0
    seed: int | None = 42
