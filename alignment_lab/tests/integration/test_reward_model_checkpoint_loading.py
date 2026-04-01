from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
from peft import get_peft_model
from transformers import AutoModelForSequenceClassification, LlamaConfig, LlamaForCausalLM

from alignlab.models.factory import load_reward_model
from alignlab.models.peft_utils import build_lora_config
from alignlab.models.specs import ModelSpec


def _score_weight(module: torch.nn.Module) -> torch.Tensor:
    score_module = getattr(getattr(getattr(module, "base_model", None), "model", None), "score", None)
    if score_module is None:
        score_module = getattr(module, "score")
    modules_to_save = getattr(score_module, "modules_to_save", None)
    if modules_to_save is not None:
        return modules_to_save["default"].weight.detach().cpu()
    return score_module.weight.detach().cpu()


@pytest.mark.filterwarnings("ignore:Some weights of LlamaForSequenceClassification were not initialized")
def test_reward_model_peft_checkpoint_restores_saved_score_head(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    base_dir = tmp_path / "base_llama"
    adapter_dir = tmp_path / "rm_adapter"

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

    train_model = AutoModelForSequenceClassification.from_pretrained(str(base_dir), num_labels=1)
    spec = ModelSpec(
        hf_path=str(base_dir),
        family="llama",
        task_type="SEQ_CLS",
        use_lora=True,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        quantization=None,
        dtype="fp32",
    )
    peft_model = get_peft_model(train_model, build_lora_config(spec))

    expected_weight = torch.full_like(_score_weight(peft_model), 0.31415927)
    score_module = getattr(peft_model.base_model.model, "score")
    score_module.modules_to_save["default"].weight.data.copy_(expected_weight)
    peft_model.save_pretrained(adapter_dir)

    load_spec = ModelSpec(
        hf_path=str(adapter_dir),
        family="llama",
        task_type="SEQ_CLS",
        use_lora=False,
        quantization=None,
        dtype="fp32",
    )

    logger_name = "transformers.modeling_utils"
    with caplog.at_level(logging.WARNING, logger=logger_name):
        loaded_model = load_reward_model(load_spec)

    messages = [record.getMessage() for record in caplog.records]
    assert not any("newly initialized" in message and "score.weight" in message for message in messages)
    assert torch.allclose(_score_weight(loaded_model), expected_weight)
