"""Dependency regression: Gemma 4 must retain heterogeneous attention parameters."""

import pytest

pytest.importorskip("vllm.config")

from transformers import Gemma4UnifiedTextConfig
from transformers.integrations.heterogeneity.configuration_utils import (
    AmbiguousGlobalPerLayerAttributeError,
)
from vllm.transformers_utils.configs.gemma4 import gemma4_layer_config
from vllm.transformers_utils.model_arch_config_convertor import (
    Gemma4ModelArchConfigConvertor,
)


def test_gemma4_converter_preserves_per_layer_attention():
    config = Gemma4UnifiedTextConfig.from_dict(
        {
            "num_hidden_layers": 2,
            "layer_types": ["sliding_attention", "full_attention"],
            "head_dim": 256,
            "global_head_dim": 512,
            "num_key_value_heads": 8,
            "num_global_key_value_heads": 1,
            "attention_k_eq_v": True,
        }
    )
    with pytest.raises(AmbiguousGlobalPerLayerAttributeError):
        _ = config.head_dim
    converter = Gemma4ModelArchConfigConvertor(config, config)
    layers = converter.get_per_layer_hf_configs()
    assert layers is not None
    assert [(c.head_dim, c.num_key_value_heads) for _, c in layers] == [
        (256, 8),
        (512, 1),
    ]
    for i, (_, layer) in enumerate(layers):
        resolved = gemma4_layer_config(config, i)
        assert resolved.head_dim == layer.head_dim
        assert resolved.num_key_value_heads == layer.num_key_value_heads
    # Exercise the converter entry point that previously accessed global head_dim.
    converted = converter.convert(supports_multimodal=False)
    assert [
        (converted[i].head_size, converted[i].total_num_kv_heads) for i in range(2)
    ] == [(256, 8), (512, 1)]
