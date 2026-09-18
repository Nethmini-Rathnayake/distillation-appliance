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
