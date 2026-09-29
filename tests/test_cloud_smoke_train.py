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


def _encode_varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _field(number: int, wire: int, payload: bytes) -> bytes:
    key = _encode_varint((number << 3) | wire)
    return key + (_encode_varint(len(payload)) + payload if wire == 2 else payload)


def _event(step: int, tag: str, value: float) -> bytes:
    import struct
    val = _field(1, 2, tag.encode()) + _field(2, 5, struct.pack("<f", value))
    summary = _field(1, 2, val)
    return _field(1, 1, struct.pack("<d", 0.0)) + _field(2, 0, _encode_varint(step)) + _field(5, 2, summary)


def _write_events(path, events):
    import struct
    with open(path, "wb") as f:
        for ev in events:
            f.write(struct.pack("<Q", len(ev)) + b"\0" * 4 + ev + b"\0" * 4)


def test_steady_state_throughput_excludes_the_compile_chunk(tmp_path):
    from cloud.smoke_train import steady_state_throughput, training_timing_from_events

    _write_events(tmp_path / "events.out.tfevents.1", [
        _event(0, "eval/episode_reward", 13.0),                 # initial eval: no training metrics
        _event(819_200, "training/walltime", 400.0),            # chunk 1: includes compilation
        _event(819_200, "eval/episode_reward", 15.0),
        _event(1_638_400, "training/walltime", 440.0),          # chunk 2: 40 s of pure training
    ])
    points = training_timing_from_events(tmp_path)
    assert points == [(819_200, 400.0), (1_638_400, 440.0)]
    assert steady_state_throughput(points)["steady_training_steps_per_second"] == pytest.approx(20_480)


def test_steady_state_throughput_needs_two_training_chunks(tmp_path):
    from cloud.smoke_train import steady_state_throughput
    assert steady_state_throughput([(327_680, 284.0)])["steady_training_steps_per_second"] is None
