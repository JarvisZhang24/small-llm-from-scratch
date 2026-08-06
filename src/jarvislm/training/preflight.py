"""Pass/fail checks for a full-architecture A100 training preflight."""

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

_METRIC_LINE = re.compile(
    r"step\s+(?P<step>\d+)\s+\|\s+loss\s+(?P<loss>[-+\d.eE]+)"
    r"\s+\|\s+lr\s+[-+\d.eE]+\s+\|\s+grad\s+(?P<grad>[-+\d.eE]+)"
)


@dataclass(frozen=True)
class PreflightReport:
    passed: bool
    completed_step: int | None
    expected_step: int
    metric_records: int
    first_loss: float | None
    last_loss: float | None
    max_grad_norm: float | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_training_metrics(log_path: str | Path) -> list[tuple[int, float, float]]:
    """Parse logged ``step | loss | lr | grad`` rows from a trainer log."""
    metrics: list[tuple[int, float, float]] = []
    for line in Path(log_path).read_text().splitlines():
        match = _METRIC_LINE.search(line)
        if match is not None:
            metrics.append(
                (
                    int(match["step"]),
                    float(match["loss"]),
                    float(match["grad"]),
                )
            )
    return metrics


def assess_preflight(
    metrics: list[tuple[int, float, float]],
    *,
    completed_step: int | None,
    expected_step: int,
    min_metric_records: int = 20,
    max_grad_norm: float = 1_000.0,
) -> PreflightReport:
    """Apply the reference smoke-test checks to an A100 350M run.

    The original smoke script accepted a run when its final loss was below its
    first loss.  Here we additionally require enough logged points, finite
    gradients, a completed checkpoint, and a conservative gradient bound.
    """
    reasons: list[str] = []
    if expected_step <= 0 or min_metric_records <= 0 or max_grad_norm <= 0:
        raise ValueError("preflight thresholds must be positive")
    if completed_step is None or completed_step < expected_step:
        reasons.append(
            f"checkpoint step {completed_step!r} is below required step {expected_step}"
        )
    if len(metrics) < min_metric_records:
        reasons.append(
            f"only {len(metrics)} metric records; need at least {min_metric_records}"
        )

    losses = [loss for _, loss, _ in metrics]
    grad_norms = [grad for _, _, grad in metrics]
    if not all(math.isfinite(value) for value in losses + grad_norms):
        reasons.append("loss or gradient norm is non-finite")
    if grad_norms and (min(grad_norms) <= 0 or max(grad_norms) > max_grad_norm):
        reasons.append(f"gradient norms are outside (0, {max_grad_norm:g}]")
    if len(losses) >= 2 and losses[-1] >= losses[0]:
        reasons.append(f"loss did not decrease ({losses[0]:.4f} -> {losses[-1]:.4f})")

    return PreflightReport(
        passed=not reasons,
        completed_step=completed_step,
        expected_step=expected_step,
        metric_records=len(metrics),
        first_loss=losses[0] if losses else None,
        last_loss=losses[-1] if losses else None,
        max_grad_norm=max(grad_norms) if grad_norms else None,
        reasons=tuple(reasons),
    )


def checkpoint_step(path: str | Path) -> int:
    """Read the completed step from a JarvisLM checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "step" not in checkpoint:
        raise ValueError(f"invalid JarvisLM checkpoint: {path}")
    return int(checkpoint["step"])


__all__ = [
    "PreflightReport",
    "assess_preflight",
    "checkpoint_step",
    "parse_training_metrics",
]
