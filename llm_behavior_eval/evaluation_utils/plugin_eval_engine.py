from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, cast

import torch
from accelerate.utils import find_executable_batch_size
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, set_seed
from transformers.data.data_collator import DataCollator
from transformers.utils.quantization_config import BitsAndBytesConfig

from .eval_config import EvaluationConfig
from .eval_engine import EvalEngine
from .max_batch_size import MAX_BATCH_SIZE
try:
    # Preferred import path in production usage (external package)
    from h_core.debias.methods.plugin_model import PluginModelForCausalLM
except Exception:  # noqa: BLE001
    # Fallback for this repo (keeps local dev/tests working)
    from .plugin_model import PluginModelForCausalLM
from .sampling_config import SamplingConfig
from .util_functions import load_tokenizer_with_transformers, pick_best_dtype

if TYPE_CHECKING:
    from transformers.generation.utils import GenerationMixin


class PluginEvalEngine(EvalEngine):
    """
    EvalEngine backend for plugin models that can fuse a local base model
    with a plugin and optionally query Gemini during generation.
    """

    def __init__(
        self,
        data_collator: DataCollator,
        eval_config: EvaluationConfig,
        is_judge: bool = False,
    ) -> None:
        model_path_or_repo_id = self._get_model_path_or_repo_id(eval_config, is_judge)
        model_token = self._get_model_token(eval_config, is_judge)
        use_4bit = self._get_use_4bit(eval_config, is_judge)

        self.eval_config = eval_config
        self.is_judge = is_judge
        self.data_collator = data_collator
        self.plugin_backend = getattr(eval_config, "plugin_backend", "gemini")

        if self.plugin_backend == "gemini":
            api_key_env_var = getattr(
                eval_config, "gemini_api_key_env_var", "GOOGLE_API_KEY"
            )
            if not os.environ.get(api_key_env_var):
                raise RuntimeError(
                    f"{api_key_env_var} is not set in environment variables. "
                    "Set it to a valid Gemini API key before using the plugin backend."
                )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = pick_best_dtype(device)

        cfg = AutoConfig.from_pretrained(
            model_path_or_repo_id,
            token=model_token,
            trust_remote_code=eval_config.trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model_name_or_path,
            token=model_token,
            trust_remote_code=eval_config.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": eval_config.device_map,
            "trust_remote_code": eval_config.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if model_token is not None:
            model_kwargs["token"] = model_token

        self.model = PluginModelForCausalLM.from_pretrained(
            model_path_or_repo_id,
            **model_kwargs,
        )

    def set_dataset(self, eval_dataset: Dataset) -> None:
        self.eval_dataset = eval_dataset

    def _get_first_non_oom_batch_size(self, candidate_bs: int) -> int:
        logging.info("Trying batch size: %s", candidate_bs)
        dl = DataLoader(
            cast("torch.utils.data.Dataset", self.eval_dataset),
            batch_size=candidate_bs,
            shuffle=False,
            collate_fn=self.data_collator,
        )
        it = iter(dl)
        batch = next(it)
        input_ids = batch["test_input_ids"].to(self.model.device)
        attention_mask = batch["test_attention_mask"].to(self.model.device)
        do_sample = self._get_sample_from_config(self.eval_config, self.is_judge)
        max_new_tokens = self._get_max_new_tokens(self.eval_config, self.is_judge)
        with torch.inference_mode():
            cast("GenerationMixin", self.model).generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                backend=self.plugin_backend,
            )
        return candidate_bs

    def generate_answers(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sampling_config: SamplingConfig,
    ) -> list[str]:
        if sampling_config.do_sample is None:
            do_sample = self._get_sample_from_config(self.eval_config, self.is_judge)
        else:
            do_sample = sampling_config.do_sample
        max_new_tokens = self._get_max_new_tokens(self.eval_config, self.is_judge)
        temperature = (
            sampling_config.temperature
            if sampling_config.temperature is not None
            else (1.0 if do_sample else 0.0)
        )
        top_p = sampling_config.top_p if sampling_config.top_p is not None else 1.0
        top_k = sampling_config.top_k if sampling_config.top_k not in (None, 0) else None
        seed = sampling_config.seed
        if seed is not None:
            set_seed(seed)

        device = self.model.device
        model_input_ids = input_ids.to(device)
        model_attention = attention_mask.to(device)
        with torch.inference_mode():
            outputs = cast("GenerationMixin", self.model).generate(
                input_ids=model_input_ids,
                attention_mask=model_attention,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                backend=self.plugin_backend,
            )
        generated_tokens = outputs[:, model_input_ids.shape[1] :].detach().cpu()
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    def ensure_test_model_ready(self) -> None:
        self.model.eval()

    def get_batch_size(self) -> int:
        batch_size = self._get_batch_size_from_config(self.eval_config, self.is_judge)

        if batch_size is None:
            starting_bs = max(1, min(len(self.eval_dataset), MAX_BATCH_SIZE))
            current_bs = starting_bs

            def halve_reducer():
                nonlocal current_bs
                current_bs = max(1, current_bs // 2)
                return current_bs

            wrapper = find_executable_batch_size(
                self._get_first_non_oom_batch_size,
                starting_batch_size=starting_bs,
                reduce_batch_size_fn=halve_reducer,
            )
            batch_size = cast("int", wrapper())
            logging.info("Selected batch size: %s", batch_size)
        return batch_size

    def free_model(self) -> None:
        self.model = self.model.cpu()
        del self.model


