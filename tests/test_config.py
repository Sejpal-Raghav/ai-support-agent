from pathlib import Path

import pytest
from pydantic import ValidationError

from support_agent.config import load_settings


def test_pipeline_config_is_valid() -> None:
    settings = load_settings(Path("configs/pipeline.yaml"))
    assert settings.cohort.threads == 12000
    assert settings.cohort.split_ratios == (0.70, 0.15, 0.15)
    assert settings.policy.target_error_upper == 0.05


def test_pipeline_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: '1'\nunknown: true\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_settings(path)
