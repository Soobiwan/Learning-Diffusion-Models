from datasets import load_dataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
ds = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=42).select(range(1000))
c = 0
for ex in ds:
    inst = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    out = (ex.get("output") or "").strip()
    if inp:
        prompt = f"Instruction:\n{inst}\n\nInput:\n{inp}\n\nResponse:\n"
    else:
        prompt = f"Instruction:\n{inst}\n\nResponse:\n"
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    if 1 + len(p_ids) >= 192:
        c += 1
print(f"Items with >= 191 prompt length: {c}")
