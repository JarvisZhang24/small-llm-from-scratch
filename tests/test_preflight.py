import pytest
import torch

from jarvislm.training.preflight import (
    assess_preflight,
    checkpoint_step,
    parse_training_metrics,
)


def test_preflight_passes_reference_style_loss_and_gradient_checks() -> None:
    report = assess_preflight(
        [(10, 10.8, 5.0), (20, 10.4, 4.5), (30, 10.1, 4.1)],
        completed_step=30,
        expected_step=30,
        min_metric_records=3,
        max_grad_norm=10.0,
    )

    assert report.passed is True
    assert report.first_loss == pytest.approx(10.8)
    assert report.last_loss == pytest.approx(10.1)
    assert report.max_grad_norm == pytest.approx(5.0)


def test_preflight_rejects_unfinished_or_unstable_runs() -> None:
    report = assess_preflight(
        [(10, 10.0, 2.0), (20, 10.1, 2_000.0)],
        completed_step=20,
        expected_step=30,
        min_metric_records=3,
        max_grad_norm=100.0,
    )

    assert report.passed is False
    assert any("below required" in reason for reason in report.reasons)
    assert any("only 2" in reason for reason in report.reasons)
    assert any("gradient norms" in reason for reason in report.reasons)
    assert any("loss did not decrease" in reason for reason in report.reasons)


def test_preflight_log_parser_and_checkpoint_step(tmp_path) -> None:
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "step     10 | loss 10.8000 | lr 3.00e-05 | grad 5.00 | tok/s 100\n"
        "not a metric\n"
        "step     20 | loss 10.4000 | lr 6.00e-05 | grad 4.50 | tok/s 120\n"
    )
    checkpoint_path = tmp_path / "last.pt"
    torch.save({"step": 20}, checkpoint_path)

    assert parse_training_metrics(log_path) == [(10, 10.8, 5.0), (20, 10.4, 4.5)]
    assert checkpoint_step(checkpoint_path) == 20
