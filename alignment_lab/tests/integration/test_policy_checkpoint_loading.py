from __future__ import annotations

from pathlib import Path

import torch
from peft import get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from alignlab.models.factory import load_policy_model, load_reference_model
from alignlab.models.peft_utils import build_lora_config
from alignlab.models.specs import ModelSpec


def _first_lora_a_weight(module: torch.nn.Module) -> torch.Tensor:
    for name, parameter in module.named_parameters():
        if "lora_A" in name and "weight" in name:
            return parameter.detach().cpu()
    raise AssertionError("No LoRA A weight found in module.")


def test_causal_lm_peft_checkpoint_restores_adapter_weights_for_policy_and_reference(tmp_path: Path) -> None:
    base_dir = tmp_path / "base_llama"
    adapter_dir = tmp_path / "sft_adapter"

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    LlamaForCausalLM(config).save_pretrained(base_dir)

    train_model = LlamaForCausalLM.from_pretrained(str(base_dir))
    train_spec = ModelSpec(
        hf_path=str(base_dir),
        family="llama",
        task_type="CAUSAL_LM",
        use_lora=True,
        lora_target_modules=["q_proj", "v_proj"],
        quantization=None,
        dtype="fp32",
    )
    peft_model = get_peft_model(train_model, build_lora_config(train_spec))

    expected_weight = torch.full_like(_first_lora_a_weight(peft_model), 0.27182818)
    for name, parameter in peft_model.named_parameters():
        if "lora_A" in name and "weight" in name:
            parameter.data.copy_(expected_weight)
            break
    peft_model.save_pretrained(adapter_dir)

    load_spec = ModelSpec(
        hf_path=str(adapter_dir),
        tokenizer_path=str(base_dir),
        family="llama",
        task_type="CAUSAL_LM",
        use_lora=False,
        lora_target_modules=[],
        quantization=None,
        dtype="fp32",
        padding_side="left",
    )

    loaded_policy = load_policy_model(load_spec)
    loaded_reference = load_reference_model(load_spec)

    assert torch.allclose(_first_lora_a_weight(loaded_policy), expected_weight)
    assert torch.allclose(_first_lora_a_weight(loaded_reference), expected_weight)
    assert any(parameter.requires_grad for parameter in loaded_policy.parameters())
    assert not any(parameter.requires_grad for parameter in loaded_reference.parameters())
