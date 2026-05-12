import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from pa3.data.synthetic_part_b import CLASSES
from pa3.models.overlay_embedding import apply_logit_mask
from pa3.models.vqvae import VQVAE
from pa3.train.train_part_b_lm import setup_lm_and_overlay
from pa3.utils.checkpointing import load_checkpoint
from pa3.utils.device import amp_dtype, get_device
from pa3.utils.logging import ensure_dirs, load_config
from pa3.utils.metrics import topk_tokens
from pa3.utils.seed import set_seed


PROMPTS = [
    "Draw a 16 by 16 {c}.",
    "Generate a tiny {c} image.",
]


def _sample(masked_logits, temperature: float):
    if temperature <= 0:
        return masked_logits.argmax(-1, keepdim=True)
    probs = torch.softmax(masked_logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_code_grid(lm, tok, device, base_vocab: int, prompt: str, temperature: float = 1.0):
    prompt_ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    bos = torch.full((1, 1), tok.bos_token_id, device=device, dtype=torch.long)
    image_id = torch.full((1, 1), base_vocab, device=device, dtype=torch.long)
    cur = torch.cat([bos, prompt_ids, image_id], dim=1)
    attn = torch.ones_like(cur)
    codes = []
    first_before = None
    first_after = None
    first_top5 = None
    for _ in range(20):
        logits = lm(input_ids=cur, attention_mask=attn).logits[:, -1]
        masked = apply_logit_mask(logits, "image", base_vocab)
        if len(codes) < 16:
            masked[:, base_vocab + 1] = torch.finfo(masked.dtype).min
        if first_before is None:
            first_before = logits[0].float().cpu()
            first_after = masked[0].float().cpu()
            first_top5 = topk_tokens(masked[0], tok, 5)
        nxt = _sample(masked, temperature)
        token_id = int(nxt.item())
        if token_id == base_vocab + 1 and len(codes) >= 16:
            break
        if base_vocab + 2 <= token_id < base_vocab + 258:
            codes.append(token_id - base_vocab - 2)
        cur = torch.cat([cur, nxt], dim=1)
        attn = torch.cat([attn, torch.ones_like(nxt)], dim=1)
        if len(codes) >= 16:
            break
    while len(codes) < 16:
        codes.append(0)
    return torch.tensor(codes[:16], device=device).view(4, 4), first_before, first_after, first_top5


def save_grid(images, titles, path, nrow=6):
    rows = (len(images) + nrow - 1) // nrow
    fig, axes = plt.subplots(rows, nrow, figsize=(1.6 * nrow, 1.8 * rows))
    axes = axes.reshape(-1) if hasattr(axes, "reshape") else [axes]
    for ax in axes:
        ax.axis("off")
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img.permute(1, 2, 0).clamp(0, 1))
        axes[i].set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/part_b.yaml")
    ap.add_argument("--checkpoint", default="weights/partB_lm.pt")
    ap.add_argument("--vqvae-ckpt", default="weights/vqvae_best.pt")
    ap.add_argument("--lora-r", type=int, default=16)
    args = ap.parse_args()
    cfg = load_config(args.config); set_seed(cfg["seed"]); ensure_dirs(cfg["outputs_dir"])
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"LM checkpoint not found: {args.checkpoint}")
    device = get_device()
    vqvae = VQVAE(cfg["vqvae"]["codebook_size"], cfg["vqvae"]["beta"], cfg["vqvae"]["ema"]).to(device)
    vqvae.load_state_dict(load_checkpoint(args.vqvae_ckpt)["model"]); vqvae.eval()
    lm, tok, _, base_vocab = setup_lm_and_overlay(cfg, device, amp_dtype(), lora_r=args.lora_r)
    lm.load_state_dict(load_checkpoint(args.checkpoint, map_location=device)["lm"], strict=False)
    if hasattr(lm, "gradient_checkpointing_disable"):
        lm.gradient_checkpointing_disable()
    lm.config.use_cache = True
    lm.eval()

    images, titles, qualitative = [], [], []
    first_before = first_after = None
    for c in CLASSES:
        for template in PROMPTS:
            prompt = template.format(c=c)
            ids, before, after, top5 = generate_code_grid(lm, tok, device, base_vocab, prompt, temperature=1.0)
            ids = ids.clamp(max=cfg["vqvae"]["codebook_size"] - 1)
            decoded = vqvae.decode_codes(ids.unsqueeze(0))[0].cpu()
            images.append(decoded); titles.append(c)
            if first_before is None:
                first_before, first_after = before, after
            if len(qualitative) < 2:
                qualitative.append(f"prompt={prompt}\nTOP5_FIRST_VISUAL={top5}\nids={ids.cpu().reshape(-1).tolist()}")
    save_grid(images, titles, f"{cfg['outputs_dir']}/partB_generated_grid.png")

    sweep_images, sweep_titles = [], []
    for temp in [0.5, 1.0, 1.5]:
        for c in CLASSES:
            ids, *_ = generate_code_grid(lm, tok, device, base_vocab, PROMPTS[0].format(c=c), temperature=temp)
            ids = ids.clamp(max=cfg["vqvae"]["codebook_size"] - 1)
            sweep_images.append(vqvae.decode_codes(ids.unsqueeze(0))[0].cpu())
            sweep_titles.append(f"{c} T={temp}")
    save_grid(sweep_images, sweep_titles, f"{cfg['outputs_dir']}/partB_temperature_sweep.png", nrow=6)

    finite_after = first_after[first_after > -1e4]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].hist(first_before.numpy(), bins=80); axes[0].set_title("before mask")
    axes[1].hist(finite_after.numpy(), bins=80); axes[1].set_title("after mask")
    fig.tight_layout(); fig.savefig(f"{cfg['outputs_dir']}/partB_logit_mask_hist.png", dpi=160); plt.close(fig)

    note = [
        "Spatial coherence note: the LM predicts the 4x4 image code map as a 1D raster sequence, "
        "so horizontal neighbors are always adjacent in context but vertical neighbors are separated by row length. "
        "Failures often show row-wise discontinuities or repeated local motifs.",
        "When generations fail, compare their repeated IDs against the VQ-VAE usage histogram and cosine heatmap: "
        "dead or near-duplicate codes usually decode to washed-out patches and make the LM's token choices less spatially meaningful.",
        "",
        *qualitative,
    ]
    Path(f"{cfg['outputs_dir']}/partB_imagegen_analysis.txt").write_text("\n\n".join(note))
    print("Saved LM-generated image grids, logit-mask histogram, temperature sweep, and qualitative notes.")


if __name__ == "__main__":
    main()
