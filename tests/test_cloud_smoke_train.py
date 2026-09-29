from datetime import datetime, timedelta

import pytest

from cloud.smoke_train import checkpoint_measurement


def _checkpoint(tmp_path, when: datetime, steps: int):
    path = tmp_path / f"{when:%Y_%m_%d_%H%M%S}_{steps}"
    path.mkdir()
    return path


def test_checkpoint_measurement_reports_effective_throughput(tmp_path):
    start = datetime(2026, 9, 29, 10, 0, 0)
    _checkpoint(tmp_path, start, 0)
    _checkpoint(tmp_path, start + timedelta(seconds=400), 327_680)
    (tmp_path / "events.out.tfevents").touch()

    result = checkpoint_measurement(tmp_path)

    assert result["effective_timesteps"] == 327_680
    assert result["training_interval_seconds"] == 400
    assert result["training_steps_per_second"] == pytest.approx(819.2)


def test_checkpoint_measurement_handles_missing_checkpoints(tmp_path):
    result = checkpoint_measurement(tmp_path)
    assert result["checkpoints"] == []
    assert result["effective_timesteps"] is None
    assert result["training_steps_per_second"] is None
