import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from pa3.data.cifar_part_a import CIFARVQADataset, cache_clip_patches, vqa_text_collate
from pa3.models.connector import MLPConnector
from pa3.models.smollm_lora import apply_lora, load_frozen_clip, load_smollm, load_tokenizer
from pa3.train.train_part_a_phase2 import build_vqa_inputs
from pa3.train.train_part_a_phase1 import greedy_from_embeds
from pa3.utils.checkpointing import load_checkpoint
from pa3.utils.device import amp_dtype, get_device
from pa3.utils.logging import ensure_dirs, load_config
from pa3.utils.metrics import exact_match, grouped_accuracy, topk_tokens
from pa3.utils.seed import set_seed


@torch.no_grad()
def answer_one(lm, connector, clip, tok, ex, device, text_only=False):
    b = vqa_text_collate([ex], tok)
    if text_only:
        q = b["question_ids"].to(device)
        bos = torch.full((1, 1), tok.bos_token_id, device=device, dtype=torch.long)
        ids = torch.cat([bos, q], dim=1)
        attn = torch.ones_like(ids)
        logits = lm(input_ids=ids, attention_mask=attn).logits[0, -1]
        out = []
        cur, cur_attn = ids, attn
        for _ in range(6):
            nxt = lm(input_ids=cur, attention_mask=cur_attn).logits[:, -1].argmax(-1, keepdim=True)
            if int(nxt.item()) == tok.eos_token_id:
                break
            out.append(int(nxt.item()))
            cur = torch.cat([cur, nxt], 1)
            cur_attn = torch.cat([cur_attn, torch.ones_like(nxt)], 1)
        return tok.decode(out, skip_special_tokens=True).strip(), topk_tokens(logits, tok, 5)
    inputs, attn, _ = build_vqa_inputs(lm, connector, clip, b, device, tok)
    prefix_len = 1 + 49 + b["question_ids"].shape[1]
    logits = lm(inputs_embeds=inputs[:, :prefix_len], attention_mask=attn[:, :prefix_len]).logits[0, -1]
    pred = greedy_from_embeds(lm, tok, inputs[:, :prefix_len], attn[:, :prefix_len], max_new_tokens=6)
    return pred, topk_tokens(logits, tok, 5)


def write_phase_plot(outputs_dir: str):
    path = Path(outputs_dir) / "partA_phase_summary.jsonl"
    if path.exists():
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        labels = [r.get("phase", r.get("checkpoint", "")) for r in rows]
        ppls = [float(r.get("ppl", 0.0)) for r in rows]
        ratios = [r.get("R") for r in rows]
        plt.figure(figsize=(7, 4))
        plt.plot(labels, ppls, marker="o", label="Alpaca PPL")
        plt.ylabel("PPL")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(Path(outputs_dir) / "partA_phase_ppl.png", dpi=160)
        plt.close()
        if any(x is not None for x in ratios):
            plt.figure(figsize=(7, 4))
            plt.plot(labels, [float(x) if x is not None else 0.0 for x in ratios], marker="o")
            plt.ylabel("Forgetting ratio R")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(Path(outputs_dir) / "partA_phase_R.png", dpi=160)
            plt.close()
    eval_path = Path(outputs_dir) / "partA_phase_eval.jsonl"
    if eval_path.exists():
        eval_rows = [json.loads(x) for x in eval_path.read_text().splitlines() if x.strip()]
        plt.figure(figsize=(7, 4))
        plt.plot([r["step"] for r in eval_rows], [r["vqa_acc"] for r in eval_rows], marker="o")
        plt.xlabel("optimizer step")
        plt.ylabel("quick VQA accuracy")
        plt.tight_layout()
        plt.savefig(Path(outputs_dir) / "partA_phase_vqa_acc.png", dpi=160)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/part_a.yaml")
    ap.add_argument("--checkpoint", default="weights/partA_phase2_lambda_0.2.pt")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config); set_seed(cfg["seed"]); ensure_dirs(cfg["outputs_dir"])
    device = get_device(); tok = load_tokenizer(cfg["lm_name"])
    lm = apply_lora(load_smollm(cfg["lm_name"], device, amp_dtype()), r=args.lora_r)
    connector = MLPConnector().to(device); clip = load_frozen_clip(cfg["clip_name"], device)
    ckpt = load_checkpoint(args.checkpoint)
    connector.load_state_dict(ckpt["connector"])
    if "lm_lora" in ckpt:
        lm.load_state_dict(ckpt["lm_lora"], strict=False)
    ds = CIFARVQADataset(cfg["data_root"], False, 200, cfg["seed"], cfg["clip_name"])
    if cfg.get("cache_clip_patches", True):
        cache_clip_patches(ds, clip, device, int(cfg.get("clip_cache_batch_size", 128)))
    n = len(ds) if args.full else min(500, len(ds))
    preds, tgts, skills, classes = [], [], [], []
    text_preds = []
    qual = []
    for i in range(n):
        ex = ds[i]
        pred, top5 = answer_one(lm, connector, clip, tok, ex, device)
        text_pred, _ = answer_one(lm, connector, clip, tok, ex, device, text_only=True)
        preds.append(pred); tgts.append(ex["answer"]); skills.append(ex["skill"]); classes.append(ex["class_name"])
        text_preds.append(text_pred)
        correct = pred.strip().lower() == ex["answer"].strip().lower()
        kind = "failure"
        if correct and ex["skill"] in {"binary", "category"}:
            kind = "correct_easy"
        elif correct:
            kind = "correct_hard"
        qual.append((kind, i, ex, pred, top5))
    overall = exact_match(preds, tgts)
    rows = [["overall", overall]]
    rows += [[f"template:{k}", v] for k, v in grouped_accuracy(preds, tgts, skills).items()]
    rows += [[f"class:{k}", v] for k, v in grouped_accuracy(preds, tgts, classes).items()]
    maj = max(set(tgts), key=tgts.count)
    rows.append(["majority_baseline", exact_match([maj] * len(tgts), tgts)])
    rows.append(["text_only_baseline", exact_match(text_preds, tgts)])
    with open(f"{cfg['outputs_dir']}/partA_vqa_metrics.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    selected = []
    for kind in ["correct_easy", "correct_hard", "failure"]:
        selected.extend([x for x in qual if x[0] == kind][:2])
    if len(selected) < 6:
        selected_ids = {x[1] for x in selected}
        selected.extend([x for x in qual if x[1] not in selected_ids][:6 - len(selected)])
    qpath = Path(cfg["outputs_dir"]) / "partA_qualitative.txt"
    qpath.write_text("\n\n".join([f"type={kind}\nidx={i}\nQ={e['question']}\nGT={e['answer']}\nPRED={p}\nTOP5={t}" for kind, i, e, p, t in selected[:6]]))
    write_phase_plot(cfg["outputs_dir"])
    print(f"Part A VQA accuracy: {overall:.4f}; wrote metrics and qualitative examples.")


if __name__ == "__main__":
    main()
