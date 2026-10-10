import hashlib
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cloud"))
import launch  # noqa: E402


def test_bundle_preserves_repo_relative_paths_and_checksums(tmp_path):
    files = [launch.REPO / "cloud" / "jobs" / "neutral_gpu_pilot.sh", launch.REPO / "docs" / "gpu_neutral_pilot.md"]
    archive, manifest = launch.build_bundle(files, tmp_path)
    with tarfile.open(archive) as tar:
        tar.extractall(tmp_path / "x", filter="data")
    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        digest, rel = line.split("  ", 1)
        assert not rel.startswith("/") and hashlib.sha256((tmp_path / "x" / rel).read_bytes()).hexdigest() == digest


def test_bundle_list_rejects_paths_outside_the_repository(tmp_path):
    listing = tmp_path / "list.txt"
    listing.write_text("../outside.txt\n", encoding="utf-8")
    try:
        launch.bundle_files(listing)
    except SystemExit as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("outside path accepted")


def test_gpu_pilot_job_has_independent_time_guards():
    job = (launch.REPO / "cloud" / "jobs" / "neutral_gpu_pilot.sh").read_text(encoding="utf-8")
    assert "timeout --signal=INT" in job and "sha256sum -c" in job and "sleep 180" in job
