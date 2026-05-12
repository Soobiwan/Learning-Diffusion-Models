import argparse
import csv
import itertools
import math
import shutil
import subprocess
import sys
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from pa3.data.alpaca import AlpacaDataset, alpaca_collate
from pa3.data.synthetic_part_b import (
    PreencodedMultimodalDataset,
    SyntheticImageGenDataset,
    SyntheticVQADataset,
    tokenized_multimodal_collate,
)
from pa3.eval.eval_ppl import compute_ppl
from pa3.models.overlay_embedding import CodebookProjector, OverlayEmbedding, OverlayLMHead, transplant_codebook_to_overlay
from pa3.models.smollm_lora import apply_lora, load_smollm, load_tokenizer, trainable_parameter_report
from pa3.models.vqvae import VQVAE
from pa3.utils.checkpointing import load_checkpoint, save_checkpoint
from pa3.utils.device import amp_dtype, get_device, make_scaler
from pa3.utils.logging import append_jsonl, ensure_dirs, load_config
from pa3.utils.seed import set_seed


def setup_lm_and_overlay(cfg, device, dtype, lora_r, apply_adapter=True):
    tok = load_tokenizer(cfg["lm_name"], left_padding=True)
    base_vocab = len(tok)
    added = tok.add_tokens(["<image>", "</image>"] + [f"<vis_{i}>" for i in range(256)])
    lm = load_smollm(cfg["lm_name"], device, dtype)
    print(f"Added visual tokens: {added}; final vocab should be 49410 -> {len(tok)}")
    lm.resize_token_embeddings(len(tok))
    overlay = OverlayEmbedding(lm.get_input_embeddings(), base_vocab, 258).to(device)
    lm.set_input_embeddings(overlay)
    lm.set_output_embeddings(OverlayLMHead(lm.get_output_embeddings(), base_vocab, overlay).to(device))
    if apply_adapter:
        lm = apply_lora(lm, r=lora_r)
    return lm, tok, overlay, base_vocab


def warmup_projector(overlay, vqvae, device, epochs=2):
    proj = CodebookProjector(64, overlay.base.embedding_dim).to(device)
    opt = AdamW(proj.parameters(), lr=1e-3)
    text_norm = overlay.base.weight[:overlay.original_vocab].norm(dim=1).mean().detach()
    codes = vqvae.quantizer.codebook.detach().to(device)
    for _ in range(epochs):
        out = proj(codes)
        loss = (out.norm(dim=1).mean() - text_norm).pow(2)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        ratio = (proj(codes).norm(dim=1).mean() / text_norm).item()
        print(f"Overlay projector norm ratio visual/text: {ratio:.3f} (target [0.2, 5])")
        transplant_codebook_to_overlay(overlay, proj, vqvae.quantizer.codebook.detach())


@torch.no_grad()
def verify_and_rescale_overlay(overlay):
    text_norm = overlay.base.weight[:overlay.original_vocab].norm(dim=1).mean().clamp_min(1e-6)
    visual = overlay.overlay.weight[2:]
    ratio = (visual.norm(dim=1).mean() / text_norm).item()
    scale = 1.0
    if ratio < 0.2:
        scale = 0.2 / max(ratio, 1e-6)
    elif ratio > 5.0:
        scale = 5.0 / ratio
    if scale != 1.0:
        visual.mul_(scale)
    final_ratio = (visual.norm(dim=1).mean() / text_norm).item()
    print(f"Overlay visual/text norm ratio after transplant: {ratio:.3f}; scale={scale:.3f}; final={final_ratio:.3f}")
    return final_ratio, scale


def keep_only_special_overlay_rows_trainable(overlay):
    mask = torch.zeros_like(overlay.overlay.weight)
    mask[:2] = 1
    overlay.overlay.weight.requires_grad_(True)
    overlay.overlay.weight.register_hook(lambda grad: grad * mask)


def write_token_type_debug(vqa_ds, img_ds, out_path):
    rows = []
    for name, ds in [("vqa", vqa_ds), ("imggen", img_ds)]:
        for i in range(min(3, len(ds))):
            rows.append(f"{name}[{i}]: {' '.join(ds[i]['token_types'])}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/part_b.yaml")
    ap.add_argument("--vqvae-ckpt", default="weights/vqvae_best.pt")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--lambda-replay", type=float, default=None)
    ap.add_argument("--gamma-img", type=float, default=None)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--no-projector", action="store_true")
    ap.add_argument("--frozen-embedding", action="store_true")
    ap.add_argument("--run-ablation-table", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config); set_seed(cfg["seed"]); ensure_dirs(cfg["weights_dir"], cfg["outputs_dir"])
    if args.run_ablation_table:
        rows = [["condition", "lambda", "gamma_img", "lora_r", "checkpoint"]]
        conditions = [
            ("no_replay", 0.0, 0.0, 16),
            ("weak", 0.05, 0.05, 16),
            ("baseline", 0.2, 0.5, 16),
            ("strong", 0.5, 0.5, 16),
            ("break_protection", 0.0, 0.0, 64),
        ]
        for name, lam, gam, rank in conditions:
            subprocess.run([
                sys.executable, "-m", "pa3.train.train_part_b_lm", "--config", args.config,
                "--vqvae-ckpt", args.vqvae_ckpt, "--lambda-replay", str(lam),
                "--gamma-img", str(gam), "--lora-r", str(rank),
            ], check=True)
            ckpt = f"{cfg['weights_dir']}/partB_lm_{name}.pt"
            shutil.copyfile(f"{cfg['weights_dir']}/partB_lm.pt", ckpt)
            rows.append([name, lam, gam, rank, ckpt])
        with open(f"{cfg['outputs_dir']}/partB_ablation_table.csv", "w", newline="") as f:
            csv.writer(f).writerows(rows)
        return
    device = get_device(); dtype = amp_dtype()
    vqvae = VQVAE(cfg["vqvae"]["codebook_size"], cfg["vqvae"]["beta"], cfg["vqvae"]["ema"]).to(device)
    vqvae.load_state_dict(load_checkpoint(args.vqvae_ckpt)["model"]); vqvae.eval()
    for p in vqvae.parameters(): p.requires_grad_(False)
    lm, tok, overlay, base_vocab = setup_lm_and_overlay(cfg, device, dtype, args.lora_r, apply_adapter=False)
    ppl_n = 10 if args.debug else int(cfg.get("eval", {}).get("ppl_n", 1000))
    ppl0 = compute_ppl(lm, tok, AlpacaDataset(ppl_n, cfg["seed"]), device, cfg["max_length"], forbidden_start_id=base_vocab)
    if not args.no_projector:
        warmup_projector(overlay, vqvae, device, 1 if args.debug else 3)
    else:
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(overlay.overlay.weight[2:], a=5**0.5)
    verify_and_rescale_overlay(overlay)
    lm = apply_lora(lm, r=args.lora_r)
    if hasattr(lm, "gradient_checkpointing_enable"):
        lm.gradient_checkpointing_enable()
    lm.config.use_cache = False
    if args.frozen_embedding:
        keep_only_special_overlay_rows_trainable(overlay)
    else:
        overlay.overlay.weight.requires_grad_(True)
    trainable_parameter_report(lm)
    n_per = 20 if args.debug else cfg["vqvae"]["n_per_class"]
    vqa_base = SyntheticVQADataset("train", n_per, cfg["seed"])
    imggen_base = SyntheticImageGenDataset("train", n_per, cfg["seed"])
    vqa = PreencodedMultimodalDataset(vqa_base, tok, vqvae, device, base_vocab, "vqa", cfg["max_length"])
    imggen = PreencodedMultimodalDataset(imggen_base, tok, vqvae, device, base_vocab, "imggen", cfg["max_length"])
    write_token_type_debug(vqa, imggen, f"{cfg['outputs_dir']}/partB_token_type_debug.txt")
    vqvae.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    alp = AlpacaDataset(20 if args.debug else 1000, cfg["seed"])
    bs = 1 if args.debug else cfg["lm"]["batch_size"]
    vloader = DataLoader(vqa, batch_size=bs, shuffle=True, collate_fn=lambda b: tokenized_multimodal_collate(b, tok))
    iloader = DataLoader(imggen, batch_size=bs, shuffle=True, collate_fn=lambda b: tokenized_multimodal_collate(b, tok))
    tloader = DataLoader(alp, batch_size=bs, shuffle=True, collate_fn=lambda b: alpaca_collate(b, tok, cfg["max_length"], forbidden_start_id=base_vocab))
    opt = AdamW([
        {"params": [p for n, p in lm.named_parameters() if "lora_" in n and p.requires_grad], "lr": cfg["lm"]["lora_lr"]},
        {"params": [overlay.overlay.weight], "lr": cfg["lm"]["overlay_lr"]},
    ])
    steps_per_epoch = max(len(vloader), len(iloader), len(tloader))
    epochs = 1 if args.debug else cfg["lm"]["epochs"]
    scaler = make_scaler(); grad_accum = 1 if args.debug else cfg["lm"]["grad_accum"]
    total_optim_steps = max(1, math.ceil(steps_per_epoch * epochs / grad_accum))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[cfg["lm"]["lora_lr"], cfg["lm"]["overlay_lr"]], total_steps=total_optim_steps, pct_start=0.1)
    lam = cfg["lm"]["lambda_replay"] if args.lambda_replay is None else args.lambda_replay
    gam = cfg["lm"]["gamma_img"] if args.gamma_img is None else args.gamma_img
    vit, iit, tit = itertools.cycle(vloader), itertools.cycle(iloader), itertools.cycle(tloader)
    nan_skips = 0
    start_time = time.time()
    opt.zero_grad(set_to_none=True)
    for step in range(steps_per_epoch * epochs):
        vb, ib, tb = next(vit), next(iit), next(tit)
        tb = {k: v.to(device) for k, v in tb.items()}
        vb = {k: v.to(device) if torch.is_tensor(v) else v for k, v in vb.items() if k not in {"meta", "token_types"}}
        ib = {k: v.to(device) if torch.is_tensor(v) else v for k, v in ib.items() if k not in {"meta", "token_types"}}
        raw_losses = {}
        for name, batch, weight in [("LVQA", vb, 1.0), ("LIMG", ib, gam), ("LLM", tb, lam)]:
            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                task_loss = lm(**batch).loss
                weighted = weight * task_loss / grad_accum
            raw_losses[name] = task_loss.detach()
            if torch.isfinite(weighted):
                scaler.scale(weighted).backward()
            else:
                nan_skips += 1
        if (step + 1) % grad_accum == 0 or (step + 1) == steps_per_epoch * epochs:
            old_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if (not scaler.is_enabled()) or scaler.get_scale() >= old_scale:
                sched.step()
            opt.zero_grad(set_to_none=True)
        if step % (1 if args.debug else 25) == 0:
            total = raw_losses["LVQA"] + gam * raw_losses["LIMG"] + lam * raw_losses["LLM"]
            row = {
                "step": step,
                "LVQA": float(raw_losses["LVQA"].item()),
                "LIMG": float(raw_losses["LIMG"].item()),
                "LLM": float(raw_losses["LLM"].item()),
                "total": float(total.item()),
                "scale": float(scaler.get_scale()),
                "nan_skips": nan_skips,
            }
            print(row); append_jsonl(f"{cfg['outputs_dir']}/partB_lm_log.jsonl", row)
        if args.debug and step >= 3:
            break
    ppl = compute_ppl(lm, tok, AlpacaDataset(ppl_n, cfg["seed"]), device, cfg["max_length"], forbidden_start_id=base_vocab)
    elapsed = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    save_checkpoint(
        f"{cfg['weights_dir']}/partB_lm.pt",
        lm=lm.state_dict(),
        base_vocab=base_vocab,
        tokenizer_len=len(tok),
        ppl0=ppl0,
        ppl=ppl,
        forgetting_ratio=ppl / max(ppl0, 1e-8),
        lambda_replay=lam,
        gamma_img=gam,
        lora_r=args.lora_r,
        no_projector=args.no_projector,
        frozen_embedding=args.frozen_embedding,
        train_seconds=elapsed,
        peak_vram_gb=peak_vram,
    )
    append_jsonl(f"{cfg['outputs_dir']}/partB_lm_summary.jsonl", {
        "checkpoint": f"{cfg['weights_dir']}/partB_lm.pt",
        "lambda": lam,
        "gamma_img": gam,
        "lora_r": args.lora_r,
        "ppl0": ppl0,
        "ppl": ppl,
        "R": ppl / max(ppl0, 1e-8),
        "train_seconds": elapsed,
        "peak_vram_gb": peak_vram,
    })


if __name__ == "__main__":
    main()
