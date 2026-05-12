import argparse
import csv
import itertools
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from pa3.data.alpaca import AlpacaDataset, alpaca_collate
from pa3.data.cifar_part_a import CIFARVQADataset, cache_clip_patches, vqa_text_collate, extract_clip_patches
from pa3.eval.eval_ppl import compute_ppl
from pa3.models.connector import MLPConnector
from pa3.models.smollm_lora import apply_lora, freeze_module, load_frozen_clip, load_smollm, load_tokenizer, trainable_parameter_report
from pa3.train.train_part_a_phase1 import greedy_from_embeds
from pa3.utils.checkpointing import load_checkpoint, save_checkpoint
from pa3.utils.device import amp_dtype, get_device, make_scaler
from pa3.utils.logging import append_jsonl, ensure_dirs, load_config
from pa3.utils.metrics import exact_match
from pa3.utils.seed import set_seed


def build_vqa_inputs(lm, connector, clip, batch, device, tokenizer):
    qids, qmask = batch["question_ids"].to(device), batch["question_mask"].to(device)
    aids, amask = batch["answer_ids"].to(device), batch["answer_mask"].to(device)
    emb = lm.get_input_embeddings()
    if "clip_patches" in batch:
        patches = batch["clip_patches"].to(device=device, dtype=next(connector.parameters()).dtype)
    else:
        patches = extract_clip_patches(clip, batch["pixel_values"].to(device))
    vis = connector(patches).to(dtype=emb.weight.dtype)
    bos = torch.full((aids.shape[0], 1), tokenizer.bos_token_id, device=device, dtype=torch.long)
    inputs = torch.cat([emb(bos), vis, emb(qids), emb(aids)], 1)
    labels = torch.full(inputs.shape[:2], -100, dtype=torch.long, device=device)
    start = 1 + vis.shape[1] + qids.shape[1]
    labels[:, start:] = aids.masked_fill(amask == 0, -100)
    attn = torch.cat([torch.ones((aids.shape[0], 1 + vis.shape[1]), device=device, dtype=torch.long), qmask, amask], 1)
    return inputs, attn, labels


def norm_alignment_loss(inputs, batch, lm, device):
    qmask = batch["question_mask"].to(device)
    visual = inputs[:, 1:50]
    text = lm.get_input_embeddings()(batch["question_ids"].to(device))
    text = text[qmask.bool()]
    if text.numel() == 0:
        return inputs.new_zeros(())
    return (visual.norm(dim=-1).mean() - text.norm(dim=-1).mean()).pow(2)


@torch.no_grad()
def quick_vqa_acc(lm, connector, clip, tokenizer, ds, device, n=200):
    lm.eval(); connector.eval()
    preds, tgts = [], []
    for i in range(min(n, len(ds))):
        b = vqa_text_collate([ds[i]], tokenizer)
        inputs, attn, _ = build_vqa_inputs(lm, connector, clip, b, device, tokenizer)
        prefix_len = 1 + 49 + b["question_ids"].shape[1]
        pred = greedy_from_embeds(lm, tokenizer, inputs[:, :prefix_len], attn[:, :prefix_len], max_new_tokens=6)
        preds.append(pred.split("\n")[0].strip())
        tgts.append(b["answers"][0])
    lm.train(); connector.train()
    return exact_match(preds, tgts)


def train_for_lambda(cfg, lam, args, output_ckpt=None):
    device = get_device(); dtype = amp_dtype()
    tok = load_tokenizer(cfg["lm_name"])
    lm = apply_lora(load_smollm(cfg["lm_name"], device, dtype), r=args.lora_r)
    clip = load_frozen_clip(cfg["clip_name"], device)
    connector = MLPConnector().to(device)
    ckpt = load_checkpoint(args.connector_ckpt)
    connector.load_state_dict(ckpt["connector"])
    if "lm_lora" in ckpt:
        lm.load_state_dict(ckpt["lm_lora"], strict=False)
    freeze_module(clip)
    lm_trainable, lm_total, lm_pct = trainable_parameter_report(lm)
    connector_trainable = sum(p.numel() for p in connector.parameters() if p.requires_grad)
    per_class = 20 if args.debug else 1000
    vqa = CIFARVQADataset(cfg["data_root"], True, per_class, cfg["seed"], cfg["clip_name"])
    vqa_val = CIFARVQADataset(cfg["data_root"], False, 20 if args.debug else 200, cfg["seed"], cfg["clip_name"])
    if cfg.get("cache_clip_patches", True):
        cache_clip_patches(vqa, clip, device, int(cfg.get("clip_cache_batch_size", 128)))
        cache_clip_patches(vqa_val, clip, device, int(cfg.get("clip_cache_batch_size", 128)))
    alp = AlpacaDataset(20 if args.debug else 1000, cfg["seed"])
    vloader = DataLoader(vqa, batch_size=1 if args.debug else cfg["phase2"]["batch_size"], shuffle=True, collate_fn=lambda b: vqa_text_collate(b, tok))
    tloader = DataLoader(alp, batch_size=1 if args.debug else cfg["phase2"]["batch_size"], shuffle=True, collate_fn=lambda b: alpaca_collate(b, tok, cfg["max_length"]))
    opt = AdamW([p for p in itertools.chain(connector.parameters(), lm.parameters()) if p.requires_grad], lr=cfg["phase2"]["lr"])
    epochs = 1 if args.debug else cfg["phase2"]["epochs"]
    grad_accum = 1 if args.debug else cfg["phase2"]["grad_accum"]
    total_steps = max(1, math.ceil(len(vloader) * epochs / grad_accum))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["phase2"]["lr"], total_steps=total_steps, pct_start=0.1)
    scaler = make_scaler()
    titer = itertools.cycle(tloader)
    global_step = 0
    start_time = time.time()
    opt.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        for step, vb in enumerate(vloader):
            tb = next(titer)
            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                vi, va, vl = build_vqa_inputs(lm, connector, clip, vb, device, tok)
                lvqa = lm(inputs_embeds=vi, attention_mask=va, labels=vl).loss
                tb = {k: v.to(device) for k, v in tb.items()}
                llm = lm(**tb).loss if lam > 0 else vi.new_zeros(())
                lnorm = norm_alignment_loss(vi, vb, lm, device) if args.norm_loss_weight else vi.new_zeros(())
                loss = (lvqa + lam * llm + args.norm_loss_weight * lnorm) / grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0 or (step + 1) == len(vloader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(itertools.chain(lm.parameters(), connector.parameters()), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                if (not scaler.is_enabled()) or scaler.get_scale() >= old_scale:
                    sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
            if step % (1 if args.debug else 25) == 0:
                row = {
                    "step": global_step,
                    "lambda": lam,
                    "LVQA": float(lvqa.item()),
                    "LLM": float(llm.item()),
                    "Lnorm": float(lnorm.item()),
                    "loss": float((lvqa + lam * llm + args.norm_loss_weight * lnorm).item()),
                }
                print(row); append_jsonl(f"{cfg['outputs_dir']}/partA_phase2_log.jsonl", row)
            if (not args.debug and global_step and global_step % 300 == 0 and (step + 1) % grad_accum == 0) or (args.debug and step >= 2):
                acc = quick_vqa_acc(lm, connector, clip, tok, vqa_val, device, n=20 if args.debug else 200)
                append_jsonl(f"{cfg['outputs_dir']}/partA_phase_eval.jsonl", {"step": global_step, "lambda": lam, "vqa_acc": acc})
            if args.debug and step >= 2:
                break
        if args.debug:
            break
    ppl_n = 10 if args.debug else int(cfg.get("eval", {}).get("ppl_n", 1000))
    ppl = compute_ppl(lm, tok, AlpacaDataset(ppl_n, cfg["seed"]), device, cfg["max_length"])
    ppl0_path = Path(cfg["outputs_dir"]) / "ppl0.txt"
    ppl0 = float(ppl0_path.read_text().strip()) if ppl0_path.exists() else None
    forgetting_ratio = ppl / ppl0 if ppl0 else None
    elapsed = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    out = output_ckpt or f"{cfg['weights_dir']}/partA_phase2_lambda_{lam}.pt"
    save_checkpoint(
        out,
        connector=connector.state_dict(),
        lm_lora=lm.state_dict(),
        lambda_replay=lam,
        ppl0=ppl0,
        ppl=ppl,
        forgetting_ratio=forgetting_ratio,
        phase=getattr(args, "phase_name", "A2"),
        train_seconds=elapsed,
        peak_vram_gb=peak_vram,
        trainable_params=lm_trainable + connector_trainable,
        trainable_pct_lm=lm_pct,
    )
    append_jsonl(f"{cfg['outputs_dir']}/partA_phase_summary.jsonl", {
        "checkpoint": out,
        "phase": getattr(args, "phase_name", "A2"),
        "lambda": lam,
        "ppl0": ppl0,
        "ppl": ppl,
        "R": forgetting_ratio,
        "train_seconds": elapsed,
        "peak_vram_gb": peak_vram,
        "trainable_params": lm_trainable + connector_trainable,
        "lm_total_params": lm_total,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/part_a.yaml")
    ap.add_argument("--connector-ckpt", default="weights/connector_phaseA1.pt")
    ap.add_argument("--lambda-replay", type=float, default=None)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--norm-loss-weight", type=float, default=0.0)
    ap.add_argument("--output-ckpt", default=None)
    ap.add_argument("--run-ablation-table", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config); set_seed(cfg["seed"]); ensure_dirs(cfg["weights_dir"], cfg["outputs_dir"])
    if args.run_ablation_table:
        rows = [["condition", "lora_r", "checkpoint"]]
        for rank in [2, 4, 16, 32]:
            subprocess.run([
                sys.executable, "-m", "pa3.train.train_part_a_phase2", "--config", args.config,
                "--connector-ckpt", args.connector_ckpt, "--lambda-replay", str(cfg["phase2"]["lambda_replay"]),
                "--lora-r", str(rank),
            ], check=True)
            src = f"{cfg['weights_dir']}/partA_phase2_lambda_{cfg['phase2']['lambda_replay']}.pt"
            dst = f"{cfg['weights_dir']}/partA_ablation_lora_r{rank}.pt"
            shutil.copyfile(src, dst)
            rows.append([f"lora_r_{rank}", rank, dst])
        with open(f"{cfg['outputs_dir']}/partA_lora_rank_ablation.csv", "w", newline="") as f:
            csv.writer(f).writerows(rows)
        return
    vals = cfg["phase2"]["lambda_sweep"] if args.sweep else [args.lambda_replay if args.lambda_replay is not None else cfg["phase2"]["lambda_replay"]]
    for lam in vals:
        train_for_lambda(cfg, float(lam), args, output_ckpt=args.output_ckpt)


if __name__ == "__main__":
    main()
