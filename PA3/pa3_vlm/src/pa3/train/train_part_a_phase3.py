import argparse

from pa3.train.train_part_a_phase2 import train_for_lambda
from pa3.utils.logging import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/part_a.yaml")
    ap.add_argument("--connector-ckpt", default="weights/partA_phase2_lambda_0.2.pt")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--unfreeze-last-clip-blocks", type=int, default=0)
    ap.add_argument("--norm-loss-weight", type=float, default=0.0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    args.phase_name = "A3"
    cfg = load_config(args.config)
    cfg["phase2"]["lambda_replay"] = 0.0
    cfg["phase2"]["lr"] = cfg["phase3"]["lr"]
    cfg["phase2"]["epochs"] = cfg["phase3"]["epochs"]
    train_for_lambda(cfg, 0.0, args, output_ckpt=f"{cfg['weights_dir']}/connector_phaseA3.pt")


if __name__ == "__main__":
    main()
