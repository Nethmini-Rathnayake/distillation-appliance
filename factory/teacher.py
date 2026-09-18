"""Teacher model: local inference and soft-logit capture.

Responsibilities:
    - Load the teacher checkpoint locally (no hosted API; arms C and D need
      reproducible logits).
    - Arm A: few-shot prompted evaluation of the teacher itself.
    - Emit per-example soft logits over the train/val split for arm D to consume.
    - Gate: if teacher accuracy falls below `teacher.accuracy_floor`, stop and
      report rather than distilling from a bad teacher.
"""
