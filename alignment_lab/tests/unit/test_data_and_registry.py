from __future__ import annotations

from alignlab.data.adapters.gsm8k import extract_numeric_answer
from alignlab.data.adapters.hh_rlhf import HHRLHFAdapter
from alignlab.data.collators import PreferenceCollator, SFTCollator, build_prompt_response_features
from alignlab.data.loaders import get_adapter, list_adapters
from alignlab.data.schemas import PreferenceExample, SFTExample


def test_hh_parser_extracts_shared_prompt_and_responses() -> None:
    adapter = HHRLHFAdapter()
    raw = {
        "chosen": "Human: Hello\n\nAssistant: Hi there",
        "rejected": "Human: Hello\n\nAssistant: Go away",
    }
    example = adapter.raw_to_canonical(raw)
    assert example.prompt == "Human: Hello\n\nAssistant:"
    assert example.chosen == "Hi there"
    assert example.rejected == "Go away"


def test_prompt_masking_marks_only_response_tokens(tokenizer) -> None:
    features = build_prompt_response_features(
        tokenizer=tokenizer,
        prompt="Human: hello Assistant:",
        response="good",
        max_length=8,
    )
    assert features["labels"][: features["prompt_length"]] == [-100] * features["prompt_length"]
    assert features["labels"][-2:] == [tokenizer.encode("good")[0], tokenizer.eos_token_id]


def test_collator_shapes(tokenizer) -> None:
    sft_collator = SFTCollator(tokenizer=tokenizer, max_length=8)
    pref_collator = PreferenceCollator(tokenizer=tokenizer, max_length=8)

    sft_batch = sft_collator([SFTExample(prompt="Human: hello Assistant:", response="good")])
    pref_batch = pref_collator(
        [PreferenceExample(prompt="Human: hello Assistant:", chosen="good", rejected="bad")]
    )

    assert sft_batch["input_ids"].shape == sft_batch["labels"].shape
    assert pref_batch["chosen_input_ids"].shape == pref_batch["chosen_labels"].shape
    assert pref_batch["rejected_input_ids"].shape == pref_batch["rejected_labels"].shape


def test_answer_extractor_handles_multiple_formats() -> None:
    assert extract_numeric_answer("#### 1,234") == "1234"
    assert extract_numeric_answer("The answer is -12.5") == "-12.5"
    assert extract_numeric_answer("We compute... answer: 42") == "42"
    assert extract_numeric_answer("final number is 7 and that's it") == "7"


def test_registry_loading() -> None:
    names = list_adapters()
    assert "hh_rlhf" in names
    assert "gsm8k" in names
    assert get_adapter("hh_rlhf").name == "hh_rlhf"
