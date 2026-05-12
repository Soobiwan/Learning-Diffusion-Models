import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from pa3.data.cifar_part_a import CIFARCaptionDataset, cache_clip_patches, caption_collate, extract_clip_patches
from pa3.models.connector import MLPConnector, rescale_if_needed
from pa3.models.smollm_lora import load_frozen_clip, load_smollm, load_tokenizer, freeze_module
from pa3.utils.checkpointing import save_checkpoint
from pa3.utils.device import get_device, amp_dtype, make_scaler
from pa3.utils.logging import ensure_dirs, load_config
from pa3.utils.seed import set_seed


def build_inputs(lm, connector, clip, batch, device, tokenizer):
    cap_ids = batch["caption_ids"].to(device)
    cap_mask = batch["caption_mask"].to(device)
    if "clip_patches" in batch:
        patches = batch["clip_patches"].to(device=device, dtype=next(connector.parameters()).dtype)
    else:
        patches = extract_clip_patches(clip, batch["pixel_values"].to(device))
    emb = lm.get_input_embeddings()
    vis = connector(patches).to(dtype=emb.weight.dtype)
    bos = torch.full((cap_ids.shape[0], 1), tokenizer.bos_token_id, device=device, dtype=torch.long)
    bos_e = emb(bos)
    cap_e = emb(cap_ids)
    inputs = torch.cat([bos_e, vis, cap_e], dim=1)
    labels = torch.full(inputs.shape[:2], -100, dtype=torch.long, device=device)
    labels[:, 1 + vis.shape[1]:] = cap_ids.masked_fill(cap_mask == 0, -100)
    attn = torch.cat([torch.ones((cap_ids.shape[0], 1 + vis.shape[1]), device=device, dtype=torch.long), cap_mask.to(device)], dim=1)
    vis, ratio, scale = rescale_if_needed(vis, cap_e[cap_mask.bool()])
    inputs[:, 1:1 + vis.shape[1]] = vis
    return inputs, attn, labels, ratio, scale


def build_visual_prefix(lm, connector, clip, batch, device, tokenizer):
    if "clip_patches" in batch:
        patches = batch["clip_patches"].to(device=device, dtype=next(connector.parameters()).dtype)
    else:
        patches = extract_clip_patches(clip, batch["pixel_values"].to(device))
    emb = lm.get_input_embeddings()
    vis = connector(patches).to(dtype=emb.weight.dtype)
    bos = torch.full((vis.shape[0], 1), tokenizer.bos_token_id, device=device, dtype=torch.long)
    inputs = torch.cat([emb(bos), vis], dim=1)
    attn = torch.ones(inputs.shape[:2], device=device, dtype=torch.long)
    return inputs, attn


@torch.no_grad()
def scale_connector_output(connector, scale: float) -> None:
    if abs(scale - 1.0) < 1e-6:
        return
    final = connector.net[-1]
    final.weight.mul_(scale)
    final.bias.mul_(scale)


@torch.no_grad()
def greedy_from_embeds(lm, tokenizer, prefix_embeds, prefix_attn, max_new_tokens=24):
    emb = lm.get_input_embeddings()
    inputs = prefix_embeds
    attn = prefix_attn
    out = []
    for _ in range(max_new_tokens):
        logits = lm(inputs_embeds=inputs, attention_mask=attn).logits[:, -1]
        nxt = logits.argmax(-1, keepdim=True)
        token_id = int(nxt[0].item())
        if token_id == tokenizer.eos_token_id:
            break
        out.append(token_id)
        inputs = torch.cat([inputs, emb(nxt)], dim=1)
        attn = torch.cat([attn, torch.ones_like(nxt)], dim=1)
    return tokenizer.decode(out, skip_special_tokens=True).strip()


@torch.no_grad()
def greedy_caption(lm, connector, clip, tokenizer, ds, device, out_path):
    lm.eval(); connector.eval()
    rows = []
    for i in range(min(5, len(ds))):
        b = caption_collate([ds[i]], tokenizer)
        inputs, attn = build_visual_prefix(lm, connector, clip, b, device, tokenizer)
        gen = greedy_from_embeds(lm, tokenizer, inputs, attn, max_new_tokens=24)
        rows.append(f"GT: {b['captions'][0]}\nGEN: {gen}\n")
    Path(out_path).write_text("\n".join(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/part_a.yaml")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    ensure_dirs(cfg["weights_dir"], cfg["outputs_dir"])
    device = get_device()
    tok = load_tokenizer(cfg["lm_name"])
    lm = load_smollm(cfg["lm_name"], device, amp_dtype())
    clip = load_frozen_clip(cfg["clip_name"], device)
    freeze_module(lm)
    connector = MLPConnector().to(device)
    per_class = 10 if args.debug else 1000
    ds = CIFARCaptionDataset(cfg["data_root"], True, per_class, cfg["seed"], cfg["clip_name"])
    if args.limit:
        ds.indices = ds.indices[:args.limit]
    if cfg.get("cache_clip_patches", True):
        cache_clip_patches(ds, clip, device, int(cfg.get("clip_cache_batch_size", 128)))
    loader = DataLoader(ds, batch_size=2 if args.debug else cfg["phase1"]["batch_size"], shuffle=True, collate_fn=lambda b: caption_collate(b, tok))
    opt = Adam(connector.parameters(), lr=cfg["phase1"]["lr"])
    scaler = make_scaler()
    dtype = amp_dtype()
    start_time = time.time()
    for epoch in range(1 if args.debug else cfg["phase1"]["epochs"]):
        for step, batch in enumerate(loader):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                inputs, attn, labels, ratio, scale = build_inputs(lm, connector, clip, batch, device, tok)
                loss = lm(inputs_embeds=inputs, attention_mask=attn, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(connector.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if step % 10 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f} rnorm={ratio:.3f} scale={scale:.3f}")
            if args.debug and step >= 2:
                break
    with torch.no_grad():
        probe = next(iter(loader))
        _, _, _, ratio, scale = build_inputs(lm, connector, clip, probe, device, tok)
        scale_connector_output(connector, scale)
        print(f"Final Phase 1 connector rnorm={ratio:.3f}; persistent_scale={scale:.3f}")
    elapsed = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    save_checkpoint(
        f"{cfg['weights_dir']}/connector_phaseA1.pt",
        connector=connector.state_dict(),
        phase="A1",
        train_seconds=elapsed,
        peak_vram_gb=peak_vram,
        trainable_params=sum(p.numel() for p in connector.parameters() if p.requires_grad),
    )
    heldout = CIFARCaptionDataset(cfg["data_root"], False, 200 if not args.debug else 5, cfg["seed"], cfg["clip_name"])
    if cfg.get("cache_clip_patches", True):
        cache_clip_patches(heldout, clip, device, int(cfg.get("clip_cache_batch_size", 128)))
    greedy_caption(lm, connector, clip, tok, heldout, device, f"{cfg['outputs_dir']}/partA_phase1_captions.txt")


if __name__ == "__main__":
    main()
