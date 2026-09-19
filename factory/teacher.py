"""Teacher model: local inference and soft-logit capture.

Responsibilities:
    - Load the teacher checkpoint locally (no hosted API; arms C and D need
      reproducible logits).
    - Arm A: few-shot prompted evaluation of the teacher itself.
    - Emit per-example soft logits over the train/val split for arm D to consume.
    - Gate: if teacher accuracy falls below `teacher.accuracy_floor`, stop and
      report rather than distilling from a bad teacher.

Status: soft-logit capture is implemented (fine-tune teacher, gate on the
*validation* split, dump logits). Arm A few-shot prompting is not yet.

    python -m factory.teacher --config configs/arm_d.yaml
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from factory import profile as hwprofile
from factory.cli import load_config
from factory.data import load_splits
from factory.train import fit, predict_logits, set_seed, tokenize

log = logging.getLogger(__name__)


class TeacherBelowFloorError(RuntimeError):
    """Teacher validation accuracy is under teacher.accuracy_floor."""


def generate_logits(config: Mapping[str, Any]) -> Path:
    import pandas as pd

    device = hwprofile.require_device(config)
    set_seed(int(config["seed"]))
    splits = load_splits(config, cache_dir=config["paths"]["model_cache"])

    # Teacher is trained with the hard loss on the train split only.
    model, tokenizer = fit(config, config["teacher"]["checkpoint"], splits, {"kind": "hard"}, device)

    text_f, label_f = config["dataset"]["text_field"], config["dataset"]["label_field"]
    bs, tok = config["train"]["eval_batch_size"], config["tokenizer"]
    val_logits = predict_logits(model, tokenize(tokenizer, splits.val[text_f], tok), device, bs)
    val_acc = float((val_logits.argmax(1) == np.array(splits.val[label_f])).mean())
    floor = float(config["teacher"]["accuracy_floor"])
    log.info("teacher val accuracy %.4f (floor %.2f)", val_acc, floor)
    if val_acc < floor:
        raise TeacherBelowFloorError(
            f"teacher val accuracy {val_acc:.4f} < floor {floor:.2f}. Stopping: "
            "distilling from a bad teacher cannot work."
        )

    # Logits on train (for arm D) and val only. The eval set is never used here.
    train_logits = predict_logits(model, tokenize(tokenizer, splits.train[text_f], tok), device, bs)
    rows = [
        {"split": name, "row": i, "logits": row.tolist()}
        for name, arr in (("train", train_logits), ("val", val_logits))
        for i, row in enumerate(arr)
    ]
    path = Path(config["paths"]["teacher_logits"])
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)
    log.info("wrote %d teacher logit rows to %s", len(rows), path)
    return path


def main(argv: Iterable[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Generate teacher soft logits")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    generate_logits(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
