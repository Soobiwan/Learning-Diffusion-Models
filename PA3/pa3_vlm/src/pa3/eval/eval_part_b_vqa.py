import argparse
import csv
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix

from pa3.data.synthetic_part_b import SyntheticVQADataset, CLASSES, multimodal_collate
from pa3.models.overlay_embedding import apply_logit_mask
from pa3.models.vqvae import VQVAE
from pa3.train.train_part_b_lm import setup_lm_and_overlay
from pa3.utils.checkpointing import load_checkpoint
from pa3.utils.device import amp_dtype, get_device
from pa3.utils.logging import ensure_dirs, load_config
from pa3.utils.metrics import exact_match, grouped_accuracy, topk_tokens
from pa3.utils.seed import set_seed


@torch.no_grad()
def generate_answer(lm, tok, vqvae, device, base_vocab, ex, max_new=8, text_only=False):
    if text_only:
        q = tok(ex["question"], add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        bos = torch.full((1, 1), tok.bos_token_id, device=device, dtype=torch.long)
        cur = torch.cat([bos, q], 1)
        cur_attn = torch.ones_like(cur)
    else:
        batch = multimodal_collate([ex], tok, vqvae, device, base_vocab, "vqa")
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"]
        answer_start = int((labels[0] != -100).nonzero()[0])
        cur = ids[:, :answer_start]
        cur_attn = attn[:, :answer_start]
    out = []
    first_top5 = []
    lm.eval()
    for _ in range(max_new):
        logits = lm(input_ids=cur, attention_mask=cur_attn).logits[:, -1]
        logits = apply_logit_mask(logits, "vqa_text", base_vocab)
        if not first_top5:
            first_top5 = topk_tokens(logits[0], tok, 5)
        nxt = logits.argmax(-1, keepdim=True)
        if int(nxt.item()) == tok.eos_token_id:
            break
        out.append(int(nxt.item()))
        cur = torch.cat([cur, nxt], 1)
        cur_attn = torch.cat([cur_attn, torch.ones_like(nxt)], 1)
    return tok.decode(out, skip_special_tokens=True).strip(), first_top5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/part_b.yaml")
    ap.add_argument("--checkpoint", default="weights/partB_lm.pt")
    ap.add_argument("--vqvae-ckpt", default="weights/vqvae_best.pt")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config); set_seed(cfg["seed"]); ensure_dirs(cfg["outputs_dir"])
    ds = SyntheticVQADataset("val", 20 if not args.full else cfg["vqvae"]["n_per_class"], cfg["seed"])
    n = len(ds) if args.full else min(500, len(ds))
    tgts = [ds[i]["answer"] for i in range(n)]
    majority = max(set(tgts), key=tgts.count)
    preds = [majority] * n
    if Path(args.checkpoint).exists() and Path(args.vqvae_ckpt).exists():
        device = get_device()
        vqvae = VQVAE(cfg["vqvae"]["codebook_size"], cfg["vqvae"]["beta"], cfg["vqvae"]["ema"]).to(device)
        vqvae.load_state_dict(load_checkpoint(args.vqvae_ckpt)["model"]); vqvae.eval()
        for p in vqvae.parameters(): p.requires_grad_(False)
        ckpt = load_checkpoint(args.checkpoint, map_location=device)
        lm, tok, _, base_vocab = setup_lm_and_overlay(cfg, device, amp_dtype(), lora_r=args.lora_r)
        lm.load_state_dict(ckpt["lm"], strict=False)
        if hasattr(lm, "gradient_checkpointing_disable"):
            lm.gradient_checkpointing_disable()
        lm.config.use_cache = True
        generated = [generate_answer(lm, tok, vqvae, device, base_vocab, ds[i]) for i in range(n)]
        preds = [x[0] for x in generated]
        text_preds = [generate_answer(lm, tok, vqvae, device, base_vocab, ds[i], text_only=True)[0] for i in range(n)]
        qual = []
        for i, (pred, top5) in enumerate(generated):
            correct = pred.strip().lower() == ds[i]["answer"].strip().lower()
            kind = "correct" if correct else "failure"
            if len([x for x in qual if x[0] == kind]) < 2:
                qual.append((kind, i, ds[i], pred, top5))
        Path(f"{cfg['outputs_dir']}/partB_qualitative_vqa.txt").write_text("\n\n".join([
            f"type={kind}\nidx={i}\nQ={ex['question']}\nGT={ex['answer']}\nPRED={pred}\nTOP5={top5}"
            for kind, i, ex, pred, top5 in qual[:4]
        ]))
    else:
        text_preds = [majority] * n
    skills = [ds[i]["skill"] for i in range(n)]
    classes = [ds[i]["class_name"] for i in range(n)]
    rows = [["overall_exact_match", exact_match(preds, tgts)], ["majority_baseline", exact_match([majority] * n, tgts)], ["text_only_baseline", exact_match(text_preds, tgts)]]
    rows += [[f"template:{k}", v] for k, v in grouped_accuracy(preds, tgts, skills).items()]
    rows += [[f"class:{k}", v] for k, v in grouped_accuracy(preds, tgts, classes).items()]
    shape_t = [ds[i]["answer"] for i in range(n) if ds[i]["skill"] == "shape"]
    shape_p = [preds[i] for i in range(n) if ds[i]["skill"] == "shape"]
    cm = confusion_matrix(shape_t, shape_p, labels=CLASSES)
    with open(f"{cfg['outputs_dir']}/partB_vqa_metrics.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows + [["confusion_matrix_shape"]] + cm.tolist())
    print(f"Part B VQA metrics written. Accuracy={exact_match(preds, tgts):.4f}")


if __name__ == "__main__":
    main()
