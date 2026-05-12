import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from pa3.data.synthetic_part_b import SyntheticImageDataset, generate_dataset, save_synthetic_grid
from pa3.models.vqvae import VQVAE
from pa3.utils.checkpointing import load_checkpoint, save_checkpoint
from pa3.utils.device import get_device
from pa3.utils.logging import append_jsonl, ensure_dirs, load_config
from pa3.utils.seed import set_seed


@torch.no_grad()
def save_analysis(model, loader, out_dir, device):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    selected = {}
    for batch in loader:
        for image, label in zip(batch["image"], batch["label"]):
            selected.setdefault(int(label), image)
            if len(selected) == 6:
                break
        if len(selected) == 6:
            break
    x = torch.stack([selected[i] for i in sorted(selected)]).to(device)
    recon, ids, _, _ = model(x)
    fig, axes = plt.subplots(3, 6, figsize=(9, 5))
    for i in range(x.shape[0]):
        axes[0, i].imshow(x[i].permute(1, 2, 0).cpu()); axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].permute(1, 2, 0).cpu()); axes[1, i].axis("off")
        axes[2, i].imshow(ids[i].cpu(), cmap="viridis"); axes[2, i].axis("off")
    fig.tight_layout(); fig.savefig(out / "recon_token_maps.png", dpi=160); plt.close(fig)
    counts = torch.zeros(model.quantizer.num_codes)
    for b in loader:
        _, ids, *_ = model.encode(b["image"].to(device))
        counts += torch.bincount(ids.cpu().reshape(-1), minlength=model.quantizer.num_codes)
    plt.figure(figsize=(8, 3)); plt.bar(range(len(counts)), counts.numpy()); plt.tight_layout(); plt.savefig(out / "usage_hist.png", dpi=160); plt.close()
    code = model.quantizer.codebook.detach().cpu()
    sim = torch.nn.functional.normalize(code, dim=1) @ torch.nn.functional.normalize(code, dim=1).t()
    plt.figure(figsize=(6, 5)); plt.imshow(sim, vmin=-1, vmax=1, cmap="coolwarm"); plt.colorbar(); plt.tight_layout(); plt.savefig(out / "codebook_cosine_heatmap.png", dpi=160); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/part_b.yaml")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--codebook-size", type=int, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--run-ablations", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config); set_seed(cfg["seed"]); ensure_dirs(cfg["weights_dir"], cfg["outputs_dir"])
    if args.run_ablations:
        configs = [
            ("baseline", 256, 0.25, False),
            ("K128", 128, 0.25, False),
            ("beta1", 256, 1.0, False),
            ("ema", 256, 0.25, True),
        ]
        rows = [["config", "K", "beta", "ema", "best_val_loss", "best_epoch", "checkpoint"]]
        for name, k, beta, ema in configs:
            ckpt = f"{cfg['weights_dir']}/vqvae_{name}.pt"
            import shutil
            import subprocess, sys
            subprocess.run([
                sys.executable, "-m", "pa3.train.train_vqvae", "--config", args.config,
                "--codebook-size", str(k), "--beta", str(beta), *(["--ema"] if ema else []),
            ], check=True)
            result = load_checkpoint(f"{cfg['weights_dir']}/vqvae_best.pt")
            shutil.copyfile(f"{cfg['weights_dir']}/vqvae_best.pt", ckpt)
            rows.append([name, k, beta, ema, result.get("val_loss", ""), result.get("epoch", ""), ckpt])
        with open(f"{cfg['outputs_dir']}/vqvae_ablation_results.csv", "w", newline="") as f:
            csv.writer(f).writerows(rows)
        return
    n_per = 20 if args.debug else cfg["vqvae"]["n_per_class"]
    images, labels, *_ = generate_dataset(n_per, cfg["seed"])
    save_synthetic_grid(images, labels, f"{cfg['outputs_dir']}/synthetic_grid.png")
    train = SyntheticImageDataset("train", n_per, cfg["seed"]); val = SyntheticImageDataset("val", n_per, cfg["seed"])
    loader = DataLoader(train, batch_size=8 if args.debug else cfg["vqvae"]["batch_size"], shuffle=True)
    vloader = DataLoader(val, batch_size=8 if args.debug else cfg["vqvae"]["batch_size"])
    device = get_device()
    codebook_size = args.codebook_size or cfg["vqvae"]["codebook_size"]
    beta = args.beta or cfg["vqvae"]["beta"]
    ema = args.ema or cfg["vqvae"]["ema"]
    model = VQVAE(codebook_size, beta, ema).to(device)
    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=cfg["vqvae"]["lr"])
    best = 1e9
    best_stats = {}
    epochs = 2 if args.debug else cfg["vqvae"]["epochs"]
    for ep in range(epochs):
        model.train()
        for step, batch in enumerate(loader):
            x = batch["image"].to(device)
            recon, ids, loss, stats = model(x)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if step % 25 == 0:
                row = {"epoch": ep, "step": step, "loss": float(loss.item()), **stats}
                print(row); append_jsonl(f"{cfg['outputs_dir']}/vqvae_log.jsonl", row)
            if args.debug and step >= 3:
                break
        model.eval(); vals = []
        with torch.no_grad():
            for b in vloader:
                _, _, loss, _ = model(b["image"].to(device)); vals.append(loss.item())
        val_loss = sum(vals) / len(vals)
        if val_loss < best:
            best = val_loss
            best_stats = {"val_loss": val_loss, "epoch": ep}
            save_checkpoint(f"{cfg['weights_dir']}/vqvae_best.pt", model=model.state_dict(), cfg=cfg, codebook_size=codebook_size, beta=beta, ema=ema, **best_stats)
    save_analysis(model, vloader, f"{cfg['outputs_dir']}/part_b_vqvae", device)
    with open(f"{cfg['outputs_dir']}/vqvae_summary.csv", "w", newline="") as f:
        csv.writer(f).writerows([
            ["K", codebook_size],
            ["beta", beta],
            ["ema", ema],
            ["best_val_loss", best],
            ["best_epoch", best_stats.get("epoch", "")],
        ])


if __name__ == "__main__":
    main()
