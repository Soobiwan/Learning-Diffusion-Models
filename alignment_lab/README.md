# alignment-lab

`alignment-lab` is a modular research codebase for small-scale LLM alignment work on a single GPU, with an initial focus on a single RTX 2080 Ti. It is designed as a mini-framework rather than a collection of one-off scripts.

## Supported paths

- Supervised fine-tuning (SFT)
- Reward model (RM) training
- Direct Preference Optimization (DPO)
- Proximal Policy Optimization (PPO)
- Group Relative Policy Optimization (GRPO)
- Reinforcement Learning with Verifiable Rewards (RLVR)

The repository is intentionally structured so that:

- new datasets mostly require a new adapter in `src/alignlab/data/adapters/`
- new pairwise methods mostly require a new objective in `src/alignlab/objectives/`
- new model families mostly require a new model config plus tokenizer/prompt quirks

## Design goals

- package-style layout under `src/alignlab/`
- config-driven experiments under `configs/`
- conservative defaults for memory-limited hardware
- reusable trainers and rollout code
- explicit canonical data schemas
- unit and integration tests from day one

## Quick start

```bash
/usr/bin/python3.11 -m venv .venv
. .venv/bin/activate
cp .env.example .env
mkdir -p .hf_cache/datasets .hf_cache/transformers
python -m pip install --upgrade pip setuptools wheel
pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
pip install -e .
python scripts/check_imports.py
pytest tests/unit -q
```

The repository currently runs best with Python 3.11 on Linux. The default system `python3`
on some machines may be too new for the PyTorch stack used here.

If you are setting this up on a CPU-only machine, use `pip install -r requirements.txt`
instead of the CUDA wheel index above.

## Example commands

```bash
python -m alignlab.cli.train_sft --config configs/experiment/sft_hh_rlhf.yaml --dry-run
python -m alignlab.cli.train_rm --config configs/experiment/rm_hh_rlhf.yaml --dry-run
python -m alignlab.cli.train_pairwise --config configs/experiment/dpo_hh_rlhf.yaml --dry-run
python -m alignlab.cli.train_online --config configs/experiment/ppo_hh_rlhf.yaml --dry-run
python -m alignlab.cli.train_online --config configs/experiment/rlvr_gsm8k.yaml --dry-run
python -m alignlab.cli.setup_audit --config configs/experiment/pa2_ppo_hh_rlhf.yaml --dry-run
python -m alignlab.cli.compare_pa2 --dry-run
bash scripts/run_pa2_pipeline.sh smoke
```

## PA2 docs

- `docs/pa2/implementation_status.md` tracks the C0-C8 coding checklist against code and artifacts.
- `docs/pa2/implementation_guide_viva.md` explains the implementation, caveats, and likely viva-style questions.

The default PA2 execution path now reuses the archived completed reward-model checkpoint at
`artifacts/checkpoints/rm_hh_rlhf/final` and then trains the policy-side methods under the
`pa2_*` configs.

## Notes

- Reward models use `AutoModelForSequenceClassification`.
- Alignment logic does not depend on TRL, `Trainer`, or prebuilt PPO/DPO trainers.
- Frozen reference or reward models can be quantized through config when memory is tight.
- Default policy experiments target `HuggingFaceTB/SmolLM2-360M-Instruct`.
- Default RM and PPO value backbones target `meta-llama/Llama-3.2-1B-Instruct`.
- `train_rm` still exists and saves a concrete checkpoint under `artifacts/checkpoints/<experiment_name>/final`, but the default PA2 runner now reuses the archived `rm_hh_rlhf` checkpoint instead of retraining RM.
- `scripts/run_pa2_pipeline.sh` writes per-stage terminal logs to `artifacts/run_logs/<timestamp>/` and sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` unless you already provided it.
- The PA2 configs now combine subset caps, save-best checkpoints, and early stopping with hardware-fit online budgets for a 10.7 GB GPU; see `docs/pa2/implementation_guide_viva.md` for the exact values and rationale.
- Several method and dataset extensions are scaffolded with specific TODO markers for incremental follow-up.
