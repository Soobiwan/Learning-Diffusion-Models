import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA

from pa3.data.cifar_part_a import CIFARVQADataset, cache_clip_patches, vqa_text_collate, extract_clip_patches
from pa3.models.connector import MLPConnector
from pa3.models.smollm_lora import load_frozen_clip, load_smollm, load_tokenizer
from pa3.utils.checkpointing import load_checkpoint
from pa3.utils.device import amp_dtype, get_device
from pa3.utils.logging import ensure_dirs, load_config
from pa3.utils.seed import set_seed


def parse_checkpoint_specs(specs):
    out = []
    for spec in specs:
        if ":" in spec:
            label, path = spec.split(":", 1)
        else:
            path = spec
            label = Path(path).stem
        out.append((label, path))
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/part_a.yaml")
    ap.add_argument("--checkpoint", action="append", default=None, help="Use label:path or path. Can be repeated.")
    args = ap.parse_args()
    cfg = load_config(args.config); set_seed(cfg["seed"]); ensure_dirs(cfg["outputs_dir"])
    device = get_device(); tok = load_tokenizer(cfg["lm_name"])
    lm = load_smollm(cfg["lm_name"], device, amp_dtype()); clip = load_frozen_clip(cfg["clip_name"], device)
    ds = CIFARVQADataset(cfg["data_root"], False, 40, cfg["seed"], cfg["clip_name"])
    if cfg.get("cache_clip_patches", True):
        cache_clip_patches(ds, clip, device, int(cfg.get("clip_cache_batch_size", 128)))
    specs = args.checkpoint or [
        "A1:weights/connector_phaseA1.pt",
        "A2:weights/partA_phase2_lambda_0.2.pt",
        "A3:weights/connector_phaseA3.pt",
    ]
    rows = [["phase", "checkpoint", "MG", "within_visual", "within_text", "cross_modal"]]
    plot_rows = []
    last_embeds = None
    for label, path in parse_checkpoint_specs(specs):
        if not Path(path).exists():
            print(f"Skipping missing checkpoint: {path}")
            continue
        connector = MLPConnector().to(device)
        connector.load_state_dict(load_checkpoint(path)["connector"], strict=False)
        connector.eval()
        vs, ts = [], []
        for i in range(min(200, len(ds))):
            b = vqa_text_collate([ds[i]], tok)
            if "clip_patches" in b:
                patches = b["clip_patches"].to(device=device, dtype=next(connector.parameters()).dtype)
            else:
                patches = extract_clip_patches(clip, b["pixel_values"].to(device))
            v = connector(patches).mean(1)
            t = lm.get_input_embeddings()(b["question_ids"].to(device)).mean(1)
            vs.append(v.cpu()); ts.append(t.float().cpu())
        V, T = torch.cat(vs), torch.cat(ts)
        Vn, Tn = torch.nn.functional.normalize(V, dim=1), torch.nn.functional.normalize(T, dim=1)
        mg = (Vn.mean(0) - Tn.mean(0)).norm().item()
        wv = (Vn @ Vn.t()).mean().item(); wt = (Tn @ Tn.t()).mean().item(); cross = (Vn @ Tn.t()).mean().item()
        rows.append([label, path, mg, wv, wt, cross])
        plot_rows.append((label, mg))
        last_embeds = (label, Vn, Tn)
    with open(f"{cfg['outputs_dir']}/partA_modality_gap.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    if plot_rows:
        plt.figure(figsize=(5, 3))
        plt.plot([x[0] for x in plot_rows], [x[1] for x in plot_rows], marker="o")
        plt.ylabel("MG")
        plt.tight_layout()
        plt.savefig(f"{cfg['outputs_dir']}/partA_modality_gap_phases.png", dpi=160)
        plt.close()
    if last_embeds is not None:
        label, Vn, Tn = last_embeds
        xy = PCA(n_components=2).fit_transform(torch.cat([Vn, Tn]).numpy())
        plt.figure(figsize=(5, 5))
        plt.scatter(xy[:len(Vn), 0], xy[:len(Vn), 1], label="visual", s=10)
        plt.scatter(xy[len(Vn):, 0], xy[len(Vn):, 1], label="text", s=10)
        plt.title(f"{label} PCA fallback for UMAP view")
        plt.legend(); plt.tight_layout(); plt.savefig(f"{cfg['outputs_dir']}/partA_modality_umap_or_pca.png", dpi=160)
        plt.close()
    print(f"Wrote modality-gap metrics for {len(rows) - 1} checkpoint(s).")


if __name__ == "__main__":
    main()
