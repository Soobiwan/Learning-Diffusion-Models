from __future__ import annotations

import pytest
import torch

pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from alignlab.data.collators import SFTCollator
from alignlab.data.schemas import SFTExample
from alignlab.models.policy import load_policy_bundle
from alignlab.models.specs import ModelSpec
from alignlab.trainers.sft_trainer import SFTTrainer
from tests.helpers import DummyCausalLM


def _build_local_model_dir(tmp_path):
    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, "hello": 3, "good": 4}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="</s>",
    )
    fast_tokenizer.save_pretrained(tmp_path)

    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=16,
        n_ctx=16,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    model.save_pretrained(tmp_path)
    return tmp_path


def test_model_loading_from_local_dir(tmp_path) -> None:
    model_dir = _build_local_model_dir(tmp_path)
    spec = ModelSpec(
        hf_path=str(model_dir),
        family="smollm",
        tokenizer_path=str(model_dir),
        use_lora=False,
        lora_target_modules=[],
    )
    bundle = load_policy_bundle(spec)
    assert bundle.tokenizer.pad_token_id == 0
    assert bundle.parameter_summary["total_parameters"] > 0


def test_basic_gpu_forward_if_cuda_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    model = DummyCausalLM(vocab_size=32).cuda()
    input_ids = torch.tensor([[1, 2, 3]], device="cuda")
    logits = model(input_ids=input_ids).logits
    assert logits.is_cuda


def test_tiny_end_to_end_local_sft(tmp_path) -> None:
    model_dir = _build_local_model_dir(tmp_path)
    spec = ModelSpec(
        hf_path=str(model_dir),
        family="smollm",
        tokenizer_path=str(model_dir),
        use_lora=False,
        lora_target_modules=[],
    )
    bundle = load_policy_bundle(spec)
    collator = SFTCollator(tokenizer=bundle.tokenizer, max_length=8)
    batch = collator([SFTExample(prompt="hello", response="good")])
    trainer = SFTTrainer(model=bundle.model, learning_rate=1.0e-3)
    metrics = trainer.train_batch(batch)
    assert "loss" in metrics
