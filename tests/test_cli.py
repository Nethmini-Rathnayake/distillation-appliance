from __future__ import annotations

from factory import cli


def test_load_config_resolves_extends_and_arm_settings():
    cfg = cli.load_config("configs/arm_c.yaml")

    assert cfg["arm"] == "student_hard"
    assert cfg["dataset"]["name"] == "banking77"
    assert cfg["teacher"]["checkpoint"] == "roberta-large"
    assert cfg["paths"]["runs_dir"] == "runs"
    assert "extends" not in cfg


def test_main_accepts_config_flag():
    rc = cli.main(["--config", "configs/arm_c.yaml"])
    assert rc == 0
