"""Appliance distillation prototype.

Package layout:
    data.py      dataset load, frozen eval-set carve-out, splits
    teacher.py   teacher inference and soft-logit capture
    train.py     arm training entrypoint (hard-label and distillation losses)
    evaluate.py  metrics: accuracy, macro F1, ECE, latency percentiles
    profile.py   size-on-disk, peak memory, bandwidth-adjusted throughput
    cli.py       command dispatch
"""

__version__ = "0.1.0"
