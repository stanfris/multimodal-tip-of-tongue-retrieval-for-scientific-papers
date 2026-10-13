"""Catch config API changes that prevent vLLM from constructing the judge."""
from __future__ import annotations


def test_gemma4_unified_preserves_layer_head_sizes_and_global_config_access() -> None:
    from transformers import AutoConfig

    config = AutoConfig.for_model("gemma4_unified")
    text = config.get_text_config()
    # This is the read that older vLLM performs during model arch conversion.
    head_dim = getattr(text, "head_dim", 0)
    assert head_dim > 0
    assert text.global_head_dim > 0
    assert head_dim != text.global_head_dim
    assert "full_attention" in text.layer_types
    assert "sliding_attention" in text.layer_types
