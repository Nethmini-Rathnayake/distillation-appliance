"""Training entrypoint for arms C and D.

    python -m factory.train --config configs/arm_d.yaml

Responsibilities:
    - Seed python / numpy / torch / cuda from config.
    - Build the student from `student.checkpoint`.
    - Select the loss from `loss.kind`:
        hard         -> CrossEntropy(student_logits, gold)
        distillation -> alpha * CE + (1 - alpha) * T**2 * KL(
                            log_softmax(student/T), softmax(teacher/T))
    - Write runs/<timestamp>_<arm>/{config.yaml, metrics.json, predictions.parquet}.

Rule: `loss` is the only block that may differ between arm_c and arm_d. Every
other hyperparameter is inherited from base.yaml so it cannot drift.
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from factory import profile as hwprofile
from factory.cli import load_config
from factory.data import Splits, load_splits
from factory.evaluate import compute_metrics, measure_latency, write_metrics

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_loss(student_logits, labels, loss_cfg: Mapping[str, Any], teacher_logits=None):
    """Hard-label CE, or alpha*CE + (1-alpha)*T^2*KL(student/T || teacher/T)."""
    import torch.nn.functional as F

    ce = F.cross_entropy(student_logits, labels)
    kind = loss_cfg["kind"]
    if kind == "hard":
        return ce
    if kind != "distillation":
        raise ValueError(f"unknown loss.kind: {kind!r}")
    if teacher_logits is None:
        raise ValueError("distillation loss requires teacher_logits")
    t, alpha = float(loss_cfg["temperature"]), float(loss_cfg["alpha"])
    kl = F.kl_div(
        F.log_softmax(student_logits / t, dim=-1),
        F.softmax(teacher_logits / t, dim=-1),
        reduction="batchmean",
    )
    return alpha * ce + (1.0 - alpha) * (t**2) * kl


def tokenize(tokenizer, texts: Iterable[str], tok_cfg: Mapping[str, Any]):
    return tokenizer(
        list(texts),
        max_length=tok_cfg["max_seq_length"],
        padding=tok_cfg["padding"],
        truncation=tok_cfg["truncation"],
        return_tensors="pt",
    )


def predict_logits(model, enc, device: str, batch_size: int) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, enc["input_ids"].shape[0], batch_size):
            batch = {k: v[i : i + batch_size].to(device) for k, v in enc.items()}
            out.append(model(**batch).logits.float().cpu().numpy())
    return np.concatenate(out)


def fit(
    config: Mapping[str, Any],
    checkpoint: str,
    splits: Splits,
    loss_cfg: Mapping[str, Any],
    device: str,
    teacher_logits: np.ndarray | None = None,
):
    """Fine-tune `checkpoint` on splits.train, selecting nothing on the eval set.

    Validation accuracy is logged per epoch; the frozen eval set is never touched here.
    Returns (model, tokenizer).
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler

    tcfg, tok_cfg = config["train"], config["tokenizer"]
    text_f, label_f = config["dataset"]["text_field"], config["dataset"]["label_field"]
    cache = config["paths"]["model_cache"]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=splits.num_labels, cache_dir=cache
    ).to(device)

    train_enc = tokenize(tokenizer, splits.train[text_f], tok_cfg)
    val_enc = tokenize(tokenizer, splits.val[text_f], tok_cfg)
    y_train = torch.tensor(splits.train[label_f])
    y_val = np.array(splits.val[label_f])
    t_logits = torch.tensor(teacher_logits) if teacher_logits is not None else None

    n = y_train.shape[0]
    bs = tcfg["batch_size"]
    steps_per_epoch = (n + bs - 1) // bs
    total_steps = steps_per_epoch * tcfg["epochs"]
    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for k, p in model.named_parameters() if not any(x in k for x in no_decay)],
         "weight_decay": tcfg["weight_decay"]},
        {"params": [p for k, p in model.named_parameters() if any(x in k for x in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=float(tcfg["learning_rate"]))
    sched = get_scheduler(
        tcfg["scheduler"], opt,
        num_warmup_steps=int(tcfg["warmup_ratio"] * total_steps), num_training_steps=total_steps,
    )
    use_fp16 = bool(tcfg["fp16"]) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    gen = torch.Generator().manual_seed(int(config["seed"]))

    for epoch in range(tcfg["epochs"]):
        model.train()
        perm = torch.randperm(n, generator=gen)
        running = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            batch = {k: v[idx].to(device) for k, v in train_enc.items()}
            labels = y_train[idx].to(device)
            tl = t_logits[idx].to(device) if t_logits is not None else None
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                logits = model(**batch).logits
            loss = compute_loss(logits.float(), labels, loss_cfg, tl)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += float(loss)
        val_logits = predict_logits(model, val_enc, device, tcfg["eval_batch_size"])
        val_acc = float((val_logits.argmax(1) == y_val).mean())
        log.info("epoch %d/%d  loss=%.4f  val_acc=%.4f",
                 epoch + 1, tcfg["epochs"], running / steps_per_epoch, val_acc)
    return model, tokenizer


def load_teacher_logits(config: Mapping[str, Any], splits: Splits) -> np.ndarray:
    """Read teacher logits for the train split, verifying row alignment."""
    import pyarrow.parquet as pq

    path = Path(config["loss"]["teacher_logits"])
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate it first: python -m factory.teacher --config configs/arm_d.yaml"
        )
    table = pq.read_table(path).to_pandas()
    train = table[table["split"] == "train"].sort_values("row")
    if len(train) != len(splits.train):
        raise ValueError(f"teacher logits have {len(train)} train rows, expected {len(splits.train)}")
    return np.stack(train["logits"].to_numpy()).astype(np.float32)


def run(config: Mapping[str, Any]) -> Path:
    import pandas as pd
    import torch

    if not config.get("trains"):
        raise ValueError(f"arm {config['arm']!r} has trains: false; nothing to train")
    device = hwprofile.require_device(config)
    set_seed(int(config["seed"]))
    splits = load_splits(config, cache_dir=config["paths"]["model_cache"])

    loss_cfg = config["loss"]
    t_logits = load_teacher_logits(config, splits) if loss_cfg["kind"] == "distillation" else None

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model, tokenizer = fit(config, config["student"]["checkpoint"], splits, loss_cfg, device, t_logits)

    # Frozen eval set: scored once, here, at the very end.
    text_f, label_f = config["dataset"]["text_field"], config["dataset"]["label_field"]
    eval_enc = tokenize(tokenizer, splits.eval[text_f], config["tokenizer"])
    logits = predict_logits(model, eval_enc, device, config["train"]["eval_batch_size"])
    y = np.array(splits.eval[label_f])
    metrics = compute_metrics(
        logits, y, n_bins=config["metrics"]["ece_bins"],
        label_names=splits.labels, num_labels=splits.num_labels,
    )

    # Latency: one request at a time.
    inf = config["inference"]
    one = {k: v[:1].to(device) for k, v in eval_enc.items()}
    model.eval()
    with torch.no_grad():
        lat = measure_latency(lambda: model(**one), warmup=inf["warmup_batches"],
                              timed=inf["latency_samples"])
    metrics.update(lat)
    metrics.update(hwprofile.adjust_latency(lat, hwprofile.bandwidth_ratio(config), device))
    metrics["input_tokens"] = float(eval_enc["attention_mask"].sum(1).float().mean())
    metrics["peak_memory_bytes"] = hwprofile.peak_memory_bytes(device)

    run_dir = Path(config["paths"]["runs_dir"]) / f"{time.strftime('%Y%m%d-%H%M%S')}_{config['arm']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(run_dir / "model")
    metrics["model_size_bytes"] = hwprofile.size_on_disk(run_dir / "model")
    metrics["arm"] = config["arm"]

    write_metrics(metrics, run_dir)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(dict(config), sort_keys=False))
    pd.DataFrame({
        "text": list(splits.eval[text_f]), "label": y.tolist(),
        "pred": logits.argmax(1).tolist(), "logits": [row.tolist() for row in logits],
    }).to_parquet(run_dir / "predictions.parquet")
    log.info("arm %s: acc=%.4f macro_f1=%.4f ece=%.4f p50=%.2fms p95=%.2fms -> %s",
             config["arm"], metrics["accuracy"], metrics["macro_f1"], metrics["ece"],
             metrics["latency_p50_ms"], metrics["latency_p95_ms"], run_dir)
    return run_dir


def main(argv: Iterable[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train arm C or D")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
