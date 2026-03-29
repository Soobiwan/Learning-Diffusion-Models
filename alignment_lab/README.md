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
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
pip install -e .
python scripts/check_imports.py
pytest tests/unit -q
```

## Example commands

```bash
python -m alignlab.cli.train_sft --config configs/experiment/sft_hh_rlhf.yaml --dry-run
python -m alignlab.cli.train_rm --config configs/experiment/rm_hh_rlhf.yaml --dry-run
python -m alignlab.cli.train_pairwise --config configs/experiment/dpo_hh_rlhf.yaml --dry-run
python -m alignlab.cli.train_online --config configs/experiment/ppo_hh_rlhf.yaml --dry-run
python -m alignlab.cli.train_online --config configs/experiment/rlvr_gsm8k.yaml --dry-run
```

## Notes

- Reward models use `AutoModelForSequenceClassification`.
- Alignment logic does not depend on TRL, `Trainer`, or prebuilt PPO/DPO trainers.
- Frozen reference or reward models can be quantized through config when memory is tight.
- Default policy experiments target `HuggingFaceTB/SmolLM2-360M-Instruct`.
- Default RM and PPO value backbones target `meta-llama/Llama-3.2-1B-Instruct`.
- `train_rm` saves a concrete checkpoint under `artifacts/checkpoints/<experiment_name>/final`, and PPO/GRPO configs are wired to consume that RM artifact path.
- Several method and dataset extensions are scaffolded with specific TODO markers for incremental follow-up.
