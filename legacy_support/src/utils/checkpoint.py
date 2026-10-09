"""Keras checkpoint helpers that emit explicit compatibility reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_keras_weights(
    model: Any,
    checkpoint: str | Path,
    *,
    by_name: bool = True,
    skip_mismatch: bool = True,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load HDF5/Keras weights and write an auditable report.

    Keras 2 does not expose PyTorch-style missing/unexpected key objects.  The
    report therefore records the requested policy and catches incompatibility
    as an explicit error instead of pretending all tensors matched.
    """
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    report = {
        "checkpoint": str(path.resolve()),
        "by_name": by_name,
        "skip_mismatch": skip_mismatch,
        "model_name": getattr(model, "name", type(model).__name__),
        "status": "pending",
    }
    try:
        if path.suffix.lower() == ".h5":
            model.load_weights(str(path), by_name=by_name, skip_mismatch=skip_mismatch)
        else:
            model.load_weights(str(path))
        report["status"] = "loaded"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if report_path is not None:
            Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise
    if report_path is not None:
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

