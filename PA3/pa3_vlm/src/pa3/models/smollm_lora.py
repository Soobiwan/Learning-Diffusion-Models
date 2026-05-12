from typing import Iterable, Tuple

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel


def load_tokenizer(name: str, left_padding: bool = False):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left" if left_padding else "right"
    return tok


def load_smollm(name: str, device, dtype=None):
    dtype = dtype or (torch.float16 if device.type == "cuda" else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    model.to(device)
    model.config.use_cache = False
    hidden = int(model.config.hidden_size)
    vocab = int(model.config.vocab_size)
    print(f"Sanity: LM hidden size should be 960 -> {hidden}")
    print(f"Sanity: LM vocab size should be 49152 before adding tokens -> {vocab}")
    return model


def apply_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)


def trainable_parameter_report(model) -> Tuple[int, int, float]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / max(total, 1)
    print(f"Trainable parameters: {trainable:,}/{total:,} ({pct:.3f}%)")
    return trainable, total, pct


def freeze_module(module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def load_frozen_clip(name: str, device):
    clip = CLIPVisionModel.from_pretrained(name).to(device)
    freeze_module(clip)
    assert not any(p.requires_grad for p in clip.parameters())
    return clip


def set_only_named_trainable(model, name_keywords: Iterable[str]) -> None:
    kws = tuple(name_keywords)
    for name, p in model.named_parameters():
        p.requires_grad_(any(k in name for k in kws))

