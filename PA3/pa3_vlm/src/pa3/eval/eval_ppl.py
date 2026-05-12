import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pa3.data.alpaca import AlpacaDataset, alpaca_collate
from pa3.models.smollm_lora import load_smollm, load_tokenizer
from pa3.utils.device import amp_dtype, get_device
from pa3.utils.logging import load_config
from pa3.utils.seed import set_seed


@torch.no_grad()
def compute_ppl(model, tokenizer, dataset, device, max_length=192, batch_size=4, forbidden_start_id=None):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda b: alpaca_collate(b, tokenizer, max_length, forbidden_start_id))
    losses = []
    dtype = amp_dtype()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            loss = model(**batch).loss
        if torch.isfinite(loss):
            losses.append(float(loss.item()))
    return math.exp(sum(losses) / max(len(losses), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/part_a.yaml")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config); set_seed(cfg["seed"])
    device = get_device()
    tok = load_tokenizer(cfg["lm_name"])
    lm = load_smollm(cfg["lm_name"], device, amp_dtype())
    ppl = compute_ppl(lm, tok, AlpacaDataset(args.n, cfg["seed"]), device, cfg["max_length"])
    print(f"PPL0 on {args.n} Alpaca examples: {ppl:.4f}")
    out = args.out or f"{cfg.get('outputs_dir', 'outputs')}/ppl0.txt"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(f"{ppl:.8f}\n")


if __name__ == "__main__":
    main()
