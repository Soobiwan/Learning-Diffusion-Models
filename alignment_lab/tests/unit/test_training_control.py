from __future__ import annotations

from alignlab.common.training_control import EarlyStopper


def test_early_stopper_min_relative_improvement_and_patience() -> None:
    stopper = EarlyStopper(
        enabled=True,
        metric="heldout_perplexity",
        mode="min",
        min_delta=0.02,
        min_delta_mode="relative",
        patience=2,
        min_steps=2,
        save_best=True,
    )

    first = stopper.update(1, {"heldout_perplexity": 10.0})
    second = stopper.update(2, {"heldout_perplexity": 9.85})
    third = stopper.update(3, {"heldout_perplexity": 9.84})
    fourth = stopper.update(4, {"heldout_perplexity": 9.83})

    assert first.improved is True
    assert first.should_save is True
    assert second.improved is False
    assert second.should_stop is False
    assert third.bad_evals == 2
    assert third.should_stop is True
    assert fourth.should_stop is True


def test_early_stopper_max_absolute_improvement() -> None:
    stopper = EarlyStopper(
        enabled=True,
        metric="preference_accuracy",
        mode="max",
        min_delta=0.02,
        min_delta_mode="absolute",
        patience=1,
        min_steps=1,
        save_best=True,
    )

    first = stopper.update(1, {"preference_accuracy": 0.55})
    second = stopper.update(2, {"preference_accuracy": 0.56})
    third = stopper.update(3, {"preference_accuracy": 0.59})

    assert first.improved is True
    assert second.improved is False
    assert second.should_stop is True
    assert third.improved is True
    assert third.best_value == 0.59
