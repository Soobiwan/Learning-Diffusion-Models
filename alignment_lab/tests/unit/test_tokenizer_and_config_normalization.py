from __future__ import annotations

from types import SimpleNamespace

from peft import TaskType

from alignlab.models.peft_utils import build_lora_config
from alignlab.models.specs import ModelSpec
from alignlab.models.tokenizer_utils import (
    normalize_model_config_special_ids,
    normalize_special_token_id,
    normalize_tokenizer_special_ids,
)


def test_normalize_special_token_id_accepts_scalar_or_list() -> None:
    assert normalize_special_token_id(7) == 7
    assert normalize_special_token_id([9, 10]) == 9
    assert normalize_special_token_id((11, 12)) == 11
    assert normalize_special_token_id(None) is None


def test_normalize_tokenizer_special_ids_uses_scalar_eos_for_pad() -> None:
    tokenizer = SimpleNamespace(
        eos_token="</s>",
        eos_token_id=[128001, 128009],
        pad_token=None,
        pad_token_id=None,
        padding_side="right",
    )

    normalized = normalize_tokenizer_special_ids(tokenizer)

    assert normalized.eos_token_id == 128001
    assert normalized.pad_token == "</s>"
    assert normalized.pad_token_id == 128001
    assert normalized.padding_side == "right"


def test_normalize_model_config_special_ids_uses_scalar_pad_token_id() -> None:
    tokenizer = SimpleNamespace(
        eos_token_id=[128001, 128009],
        pad_token_id=128001,
    )
    config = SimpleNamespace(
        eos_token_id=[128001, 128009],
        pad_token_id=[128001, 128009],
    )

    normalized = normalize_model_config_special_ids(config, tokenizer=tokenizer)

    assert normalized.eos_token_id == 128001
    assert normalized.pad_token_id == 128001


def test_seq_cls_lora_config_saves_score_head() -> None:
    spec = ModelSpec(
        hf_path="unused",
        family="llama",
        task_type="SEQ_CLS",
        use_lora=True,
        lora_target_modules=["q_proj"],
    )

    config = build_lora_config(spec)

    assert config.task_type == TaskType.SEQ_CLS
    assert config.modules_to_save == ["score"]
