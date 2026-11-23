from pydantic_settings import BaseSettings, SettingsConfigDict


def resolve_temperature(do_sample: bool, temperature: float | None) -> float:
    """Resolve a backend-independent effective generation temperature."""
    if not do_sample:
        return 0.0
    return temperature if temperature is not None else 1.0


class SamplingConfig(BaseSettings):
    """
    Configuration for sampling.

    Args:
        do_sample: Whether to sample from the model. False forces greedy decoding.
        temperature: The temperature to use when sampling is enabled.
        top_p: The top-p value for sampling.
        top_k: The top-k value for sampling.
        seed: The seed for sampling.
    """

    model_config = SettingsConfigDict(env_prefix="bias_sampling_")

    do_sample: bool | None = None
    temperature: float | None = None
    top_p: float | None = 1.0
    top_k: int | None = 0
    seed: int | None = 42
