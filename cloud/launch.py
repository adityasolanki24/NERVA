"""NERVA cloud runner: one capped, self-deleting GPU VM per job on Google Cloud.

Subcommands (anything that spends money or changes permissions prints its plan and
requires --yes):
  setup            one-time: results bucket + a dedicated VM service account
  launch --job J   upload the committed code snapshot, start a VM that runs cloud/jobs/J.sh
  status [RUN]     list NERVA VMs; with RUN, show the tail of that run's log
  fetch RUN        download a run's results to experiments/cloud_runs/RUN/
  kill RUN         delete a run's VM now

Cost safety (docs/style_policy_design.md §6): every VM gets --max-run-duration and
--instance-termination-action=DELETE (hard cap, enforced by Google), and the VM deletes
itself when its job ends (cloud/vm_startup.sh). The "nerva" budget alert is a backstop.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT = "nerva-adityapersonal"
REGION = "us-central1"
DEFAULT_ZONE = "us-central1-a"
BUCKET = f"{PROJECT}-runs"
SERVICE_ACCOUNT = f"nerva-runner@{PROJECT}.iam.gserviceaccount.com"
UPSTREAM_SHA = "b9be205ac64488c23504ca42e5ec790337adeec3"  # Open_Duck_Playground (docs/development_log.md)
IMAGE_FAMILY = "common-cu129-ubuntu-2204-nvidia-580"
IMAGE_PROJECT = "deeplearning-platform-release"
MACHINES = {"l4": "g2-standard-8", "cpu": "e2-standard-8"}  # g2 machine types include one L4
REPO = Path(__file__).resolve().parents[1]


def gcloud() -> str:
    exe = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    fallback = Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd"
    if exe:
        return exe
    if fallback.exists():
        return str(fallback)
    sys.exit("gcloud not found")


def run(args: list[str], check=True, capture=False) -> str:
    print("  $ gcloud " + " ".join(args[1:]) if args[0] == gcloud() else "  $ " + " ".join(args))
    r = subprocess.run(args, check=check, text=True, capture_output=capture)
    return (r.stdout or "").strip() if capture else ""


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *a], check=True, text=True, capture_output=True).stdout.strip()


def confirm(yes: bool, what: str) -> None:
    if not yes:
        sys.exit(f"\nNot executed. Re-run with --yes to {what}.")


# ── subcommands ──────────────────────────────────────────────────────────────

def cmd_setup(a):
    g = gcloud()
    plan = [
        [g, "storage", "buckets", "create", f"gs://{BUCKET}", f"--project={PROJECT}", f"--location={REGION}",
         "--uniform-bucket-level-access"],
        [g, "iam", "service-accounts", "create", "nerva-runner", f"--project={PROJECT}",
         "--display-name=NERVA training VMs"],
        [g, "storage", "buckets", "add-iam-policy-binding", f"gs://{BUCKET}",
         f"--member=serviceAccount:{SERVICE_ACCOUNT}", "--role=roles/storage.objectAdmin"],
        [g, "projects", "add-iam-policy-binding", PROJECT,
         f"--member=serviceAccount:{SERVICE_ACCOUNT}", "--role=roles/compute.instanceAdmin.v1",
         "--condition=None"],
    ]
    print("One-time setup plan (bucket for results; service account the VMs run as, allowed to write\n"
          "results to that bucket and to delete their own VM):")
    for p in plan:
        print("  gcloud " + " ".join(p[1:]))
    confirm(a.yes, "run this setup")
    for p in plan:
        run(p, check=False)


def cmd_launch(a):
    if git("status", "--porcelain"):
        sys.exit("Uncommitted changes: commit first so the run is tied to an exact commit.")
    sha = git("rev-parse", "HEAD")
    job = a.job
    if not (REPO / "cloud" / "jobs" / f"{job}.sh").exists():
        sys.exit(f"unknown job {job}")
    run_id = f"{job}-{time.strftime('%Y%m%d-%H%M%S')}"
    name = f"nerva-{run_id}".lower()
    g = gcloud()
    create = [
        g, "compute", "instances", "create", name, f"--project={PROJECT}", f"--zone={a.zone}",
        f"--machine-type={MACHINES[a.hw]}",
        f"--image-family={IMAGE_FAMILY}", f"--image-project={IMAGE_PROJECT}",
        "--boot-disk-size=100GB", "--maintenance-policy=TERMINATE",
        f"--service-account={SERVICE_ACCOUNT}", "--scopes=cloud-platform",
        f"--max-run-duration={a.max_hours}h", "--instance-termination-action=DELETE",
        f"--labels=nerva=1,job={job.replace('_', '-')}",
        f"--metadata-from-file=startup-script={REPO / 'cloud' / 'vm_startup.sh'}",
        "--metadata=" + ",".join([
            "install-nvidia-driver=True", f"nerva-job={job}", f"nerva-bucket={BUCKET}",
            f"nerva-run-id={run_id}", f"nerva-code-sha={sha}", f"nerva-upstream-sha={UPSTREAM_SHA}",
        ]),
    ]
    if a.spot:
        create.append("--provisioning-model=SPOT")
    print(f"Launch plan: job={job} run={run_id} code={sha[:10]} machine={MACHINES[a.hw]} "
          f"zone={a.zone} hard cap={a.max_hours} h {'SPOT' if a.spot else 'on-demand'}")
    print("  1. upload committed code snapshot:  git archive HEAD → "
          f"gs://{BUCKET}/code/{sha}.tar.gz")
    print("  2. " + "gcloud " + " ".join(create[1:]))
    confirm(a.yes, "start this VM (it costs money until it deletes itself or hits the cap)")
    archive = REPO / f".nerva-code-{sha[:10]}.tar.gz"
    subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar.gz", "-o", str(archive), "HEAD"], check=True)
    try:
        run([g, "storage", "cp", str(archive), f"gs://{BUCKET}/code/{sha}.tar.gz"])
    finally:
        archive.unlink(missing_ok=True)
    run(create)
    print(f"\nStarted. Follow with:  python cloud/launch.py status {run_id}")


def cmd_status(a):
    g = gcloud()
    run([g, "compute", "instances", "list", f"--project={PROJECT}", "--filter=labels.nerva=1",
         "--format=table(name,zone.basename(),status,creationTimestamp)"], check=False)
    if a.run:
        log = run([g, "storage", "cat", f"gs://{BUCKET}/runs/{a.run}/out/vm.log"], check=False, capture=True)
        print("\n".join(log.splitlines()[-a.lines:]) if log else "(no log synced yet; the VM syncs every 10 min)")


def cmd_fetch(a):
    dest = REPO / "experiments" / "cloud_runs" / a.run
    dest.mkdir(parents=True, exist_ok=True)
    run([gcloud(), "storage", "rsync", "--recursive", f"gs://{BUCKET}/runs/{a.run}/out", str(dest)])
    print(f"fetched to {dest}")


def cmd_kill(a):
    name = f"nerva-{a.run}".lower()
    print(f"Delete VM {name} in {a.zone} now.")
    confirm(a.yes, "delete it")
    run([gcloud(), "compute", "instances", "delete", name, f"--project={PROJECT}", f"--zone={a.zone}", "--quiet"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup"); s.add_argument("--yes", action="store_true"); s.set_defaults(f=cmd_setup)
    s = sub.add_parser("launch")
    s.add_argument("--job", required=True)
    s.add_argument("--hw", choices=sorted(MACHINES), default="l4")
    s.add_argument("--zone", default=DEFAULT_ZONE)
    s.add_argument("--max-hours", type=int, default=2)
    s.add_argument("--spot", action="store_true")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(f=cmd_launch)
    s = sub.add_parser("status"); s.add_argument("run", nargs="?"); s.add_argument("--lines", type=int, default=40)
    s.set_defaults(f=cmd_status)
    s = sub.add_parser("fetch"); s.add_argument("run"); s.set_defaults(f=cmd_fetch)
    s = sub.add_parser("kill"); s.add_argument("run"); s.add_argument("--zone", default=DEFAULT_ZONE)
    s.add_argument("--yes", action="store_true"); s.set_defaults(f=cmd_kill)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
