"""NERVA cloud runner: one capped, self-deleting GPU VM per job on Google Cloud.

Subcommands (anything that spends money or changes permissions prints its plan and
requires --yes):
  setup            one-time: results bucket + a dedicated VM service account
  network-up       create temporary Cloud NAT for private training VMs
  network-down     remove Cloud NAT when no training VMs are running
  launch --job J   upload the committed code snapshot, start a VM that runs cloud/jobs/J.sh
  status [RUN]     list NERVA VMs; with RUN, show the tail of that run's log
  fetch RUN        download a run's results to experiments/cloud_runs/RUN/
  kill RUN         delete a run's VM now
  audit            list NERVA-relevant resources that could still cost money
  teardown         delete the dedicated bucket/account/network after results are local

Cost safety (docs/style_policy_design.md §6): every VM gets --max-run-duration and
--instance-termination-action=DELETE (hard cap, enforced by Google), and the VM deletes
itself when its job ends (cloud/vm_startup.sh). The "nerva" budget alert is a backstop.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Deployment identifiers are local configuration, not repository content.
# Set NERVA_GCP_PROJECT before using any subcommand. The other values have
# non-sensitive defaults and may be overridden when needed.
PROJECT = os.environ.get("NERVA_GCP_PROJECT", "").strip()
REGION = os.environ.get("NERVA_GCP_REGION", "us-central1").strip()
DEFAULT_ZONE = os.environ.get("NERVA_GCP_ZONE", f"{REGION}-a").strip()
DEFAULT_ZONES = tuple(dict.fromkeys(
    z.strip() for z in os.environ.get(
        "NERVA_GCP_ZONES", f"{DEFAULT_ZONE},{REGION}-a,{REGION}-b,{REGION}-c"
    ).split(",") if z.strip()
))
BUCKET = os.environ.get("NERVA_GCP_BUCKET", f"{PROJECT}-runs" if PROJECT else "").strip()
SERVICE_ACCOUNT_NAME = os.environ.get("NERVA_GCP_SERVICE_ACCOUNT", "nerva-runner").strip()
SERVICE_ACCOUNT = (f"{SERVICE_ACCOUNT_NAME}@{PROJECT}.iam.gserviceaccount.com" if PROJECT else "")
ROUTER = os.environ.get("NERVA_GCP_ROUTER", "nerva-router").strip()
NAT = os.environ.get("NERVA_GCP_NAT", "nerva-nat").strip()
UPSTREAM_SHA = "b9be205ac64488c23504ca42e5ec790337adeec3"  # Open_Duck_Playground (docs/development_log.md)
IMAGE_FAMILY = "common-cu129-ubuntu-2204-nvidia-580"
IMAGE_PROJECT = "deeplearning-platform-release"
GPU_HARDWARE = {"l4", "a100"}
MACHINES = {
    "cpu": "e2-standard-8",
    "l4": "g2-standard-8",
    "a100": "a2-highgpu-1g",
}
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


def require_cloud_config() -> None:
    if not PROJECT:
        sys.exit("Set NERVA_GCP_PROJECT to your Google Cloud project ID before using cloud commands.")


def gcloud_succeeds(args: list[str]) -> bool:
    return subprocess.run(args, text=True, capture_output=True).returncode == 0


def instance_name(run_id: str) -> str:
    """Return a GCE-safe name for a run id (job names may contain underscores)."""
    return f"nerva-{run_id}".lower().replace("_", "-")


def requested_zones(a) -> list[str]:
    if getattr(a, "zone", None):
        return [a.zone]
    if getattr(a, "zones", None):
        zones = [z.strip() for z in a.zones.split(",") if z.strip()]
        if not zones:
            sys.exit("--zones must contain at least one zone")
        return list(dict.fromkeys(zones))
    return list(DEFAULT_ZONES)


def instance_zone(name: str) -> str:
    """Find a VM without requiring the caller to remember an auto-selected zone."""
    result = subprocess.run([
        gcloud(), "compute", "instances", "list", f"--project={PROJECT}",
        f"--filter=name={name}", "--format=value(zone.basename())",
    ], check=True, text=True, capture_output=True)
    zones = (result.stdout or "").strip().splitlines()
    return zones[0] if zones else ""


# ── subcommands ──────────────────────────────────────────────────────────────

def cmd_setup(a):
    require_cloud_config()
    g = gcloud()
    plan = [
        [g, "storage", "buckets", "create", f"gs://{BUCKET}", f"--project={PROJECT}", f"--location={REGION}",
         "--uniform-bucket-level-access"],
        [g, "iam", "service-accounts", "create", SERVICE_ACCOUNT_NAME, f"--project={PROJECT}",
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


def cmd_network_up(a):
    """Create the temporary egress path required by private, no-public-IP VMs."""
    require_cloud_config()
    g = gcloud()
    router_exists = gcloud_succeeds([
        g, "compute", "routers", "describe", ROUTER, f"--project={PROJECT}", f"--region={REGION}"
    ])
    nat_exists = router_exists and gcloud_succeeds([
        g, "compute", "routers", "nats", "describe", NAT, f"--router={ROUTER}",
        f"--project={PROJECT}", f"--region={REGION}"
    ])
    if nat_exists:
        print(f"Cloud NAT {NAT} on router {ROUTER} already exists.")
        return
    print("Network plan: create temporary Cloud NAT so private VMs can install dependencies.\n"
          "Cloud NAT has hourly IP/gateway and data-processing charges; run network-down after jobs finish.")
    if not router_exists:
        print(f"  gcloud compute routers create {ROUTER} --network=default --region={REGION}")
    print(f"  gcloud compute routers nats create {NAT} --router={ROUTER} --region={REGION} "
          "--auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges")
    confirm(a.yes, "create the temporary Cloud NAT")
    if not router_exists:
        run([g, "compute", "routers", "create", ROUTER, f"--project={PROJECT}",
             "--network=default", f"--region={REGION}"])
    run([g, "compute", "routers", "nats", "create", NAT, f"--router={ROUTER}",
         f"--project={PROJECT}", f"--region={REGION}", "--auto-allocate-nat-external-ips",
         "--nat-all-subnet-ip-ranges"])


def cmd_network_down(a):
    """Remove the temporary NAT and its otherwise-unused router."""
    require_cloud_config()
    g = gcloud()
    instances = run([
        g, "compute", "instances", "list", f"--project={PROJECT}", "--filter=labels.nerva=1",
        "--format=value(name)"
    ], capture=True)
    if instances:
        sys.exit("NERVA VM(s) are still present; stop or let them finish before removing Cloud NAT:\n" + instances)
    print(f"Network cleanup plan: delete NAT {NAT} and router {ROUTER} in {REGION}.")
    confirm(a.yes, "remove the temporary Cloud NAT")
    nat_exists = gcloud_succeeds([
        g, "compute", "routers", "nats", "describe", NAT, f"--router={ROUTER}",
        f"--project={PROJECT}", f"--region={REGION}"
    ])
    if nat_exists:
        run([g, "compute", "routers", "nats", "delete", NAT, f"--router={ROUTER}",
             f"--project={PROJECT}", f"--region={REGION}", "--quiet"])
    router_exists = gcloud_succeeds([
        g, "compute", "routers", "describe", ROUTER, f"--project={PROJECT}", f"--region={REGION}"
    ])
    if router_exists:
        run([g, "compute", "routers", "delete", ROUTER, f"--project={PROJECT}",
             f"--region={REGION}", "--quiet"])
    print("Temporary Cloud NAT removed.")


def cmd_launch(a):
    require_cloud_config()
    if git("status", "--porcelain"):
        sys.exit("Uncommitted changes: commit first so the run is tied to an exact commit.")
    sha = git("rev-parse", "HEAD")
    job = a.job
    if not (REPO / "cloud" / "jobs" / f"{job}.sh").exists():
        sys.exit(f"unknown job {job}")
    if not gcloud_succeeds([
        gcloud(), "compute", "routers", "nats", "describe", NAT, f"--router={ROUTER}",
        f"--project={PROJECT}", f"--region={REGION}"
    ]):
        sys.exit("Cloud NAT is not ready. Run: python cloud/launch.py network-up --yes")
    if (a.cleanup_network or a.teardown) and not a.wait:
        sys.exit("--cleanup-network and --teardown require --wait")
    if a.max_minutes is not None and a.max_minutes <= 0:
        sys.exit("--max-minutes must be positive")
    if a.max_hours is not None and a.max_hours <= 0:
        sys.exit("--max-hours must be positive")
    duration = f"{a.max_minutes}m" if a.max_minutes is not None else f"{a.max_hours}h"
    zones = requested_zones(a)
    run_id = f"{job}-{time.strftime('%Y%m%d-%H%M%S')}"
    name = instance_name(run_id)
    g = gcloud()
    input_uri = ""
    input_files: list[Path] = []
    if a.input_run:
        input_uri = f"gs://{BUCKET}/runs/{a.input_run}/out/r1/references"
        if not gcloud_succeeds([g, "storage", "ls", f"{input_uri}/styles.json"]):
            sys.exit(f"input run does not contain an R1 manifest: {input_uri}/styles.json")
    elif a.input_dir:
        input_dir = Path(a.input_dir).resolve()
        if not input_dir.is_dir():
            sys.exit(f"input directory does not exist: {input_dir}")
        manifest = input_dir / "styles.json"
        input_files = [manifest, *sorted(input_dir.glob("*.pkl"))]
        if not manifest.is_file() or len(input_files) < 2:
            sys.exit("--input-dir must contain styles.json and at least one .pkl")
        input_uri = f"gs://{BUCKET}/inputs/{run_id}"
    print(f"Launch plan: job={job} run={run_id} code={sha[:10]} machine={MACHINES[a.hw]} "
          f"zones={','.join(zones)} hard cap={duration} {'SPOT' if a.spot else 'on-demand'}")
    print("  1. upload committed code snapshot:  git archive HEAD -> "
          f"gs://{BUCKET}/code/{sha}.tar.gz")
    if input_uri:
        source = a.input_run or str(Path(a.input_dir).resolve())
        print(f"     reference input: {source} -> {input_uri}")
    print("  2. try the listed zones in order until VM creation succeeds")
    if a.wait:
        print("  3. wait for self-deletion and fetch the final results")
    if a.cleanup_network or a.teardown:
        print("  4. delete Cloud NAT/router after the VM is gone")
    if a.teardown:
        print("  5. delete the dedicated results bucket and runner service account")
    confirm(a.yes, "start this VM (it costs money until it deletes itself or hits the cap)")
    archive = REPO / f".nerva-code-{sha[:10]}.tar.gz"
    subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar.gz", "-o", str(archive), "HEAD"], check=True)
    try:
        run([g, "storage", "cp", str(archive), f"gs://{BUCKET}/code/{sha}.tar.gz"])
    finally:
        archive.unlink(missing_ok=True)
    for path in input_files:
        run([g, "storage", "cp", str(path), f"{input_uri}/{path.name}"])

    chosen_zone = ""
    failures: list[str] = []
    for zone in zones:
        create = [
            g, "compute", "instances", "create", name, f"--project={PROJECT}", f"--zone={zone}",
            f"--machine-type={MACHINES[a.hw]}",
            f"--image-family={IMAGE_FAMILY}", f"--image-project={IMAGE_PROJECT}",
            "--boot-disk-size=100GB", "--no-address",
            # GPU VMs cannot live-migrate and must TERMINATE on host maintenance; E2 CPU VMs
            # reject that policy unless preemptible, so they keep the default (MIGRATE).
            *(["--maintenance-policy=TERMINATE"] if a.hw in GPU_HARDWARE else []),
            f"--service-account={SERVICE_ACCOUNT}", "--scopes=cloud-platform",
            f"--max-run-duration={duration}", "--instance-termination-action=DELETE",
            f"--labels=nerva=1,job={job.replace('_', '-')}",
            f"--metadata-from-file=startup-script={REPO / 'cloud' / 'vm_startup.sh'}",
            "--metadata=" + ",".join(filter(None, [
                "install-nvidia-driver=True", f"nerva-job={job}", f"nerva-bucket={BUCKET}",
                f"nerva-run-id={run_id}", f"nerva-code-sha={sha}", f"nerva-upstream-sha={UPSTREAM_SHA}",
                f"nerva-input-uri={input_uri}" if input_uri else "",
            ])),
        ]
        if a.spot:
            create.append("--provisioning-model=SPOT")
        print(f"\nTrying {MACHINES[a.hw]} in {zone}...")
        result = subprocess.run(create, text=True, capture_output=True)
        if result.returncode == 0:
            if result.stdout:
                print(result.stdout.strip())
            chosen_zone = zone
            break
        detail = (result.stderr or result.stdout or "unknown gcloud error").strip().splitlines()[-1]
        failures.append(f"{zone}: {detail}")
        print(f"Unavailable in {zone}: {detail}")

    if not chosen_zone:
        if a.cleanup_network or a.teardown:
            cmd_network_down(argparse.Namespace(yes=True))
        if a.teardown:
            cmd_teardown(argparse.Namespace(yes=True, keep_network=True))
        sys.exit("VM creation failed in every requested zone:\n" + "\n".join(failures))

    print(f"\nStarted {run_id} in {chosen_zone}.")
    if not a.wait:
        print(f"Follow with:  python cloud/launch.py status {run_id}")
        return

    wait_and_finish(run_id, a)


def vm_present(name: str, attempts: int = 10) -> bool:
    """instance_zone with retries: one transient gcloud error must not abort a long wait."""
    for attempt in range(attempts):
        try:
            return bool(instance_zone(name))
        except subprocess.CalledProcessError as error:
            print(f"gcloud list failed ({error.returncode}); retry {attempt + 1}/{attempts}", file=sys.stderr)
            time.sleep(30)
    raise RuntimeError(f"could not query VM {name}; it stays under its hard cap")


def wait_and_finish(run_id: str, a) -> None:
    """Wait for self-deletion, fetch results, then remove the NAT (and optionally everything)."""
    name = instance_name(run_id)
    print("Waiting for the capped VM to finish and self-delete (Ctrl+C leaves it under its hard cap).")
    try:
        while vm_present(name):
            time.sleep(a.poll_seconds)
        print("VM is gone; fetching final output.")
        cmd_fetch(argparse.Namespace(run=run_id))
    except Exception:
        if (a.cleanup_network or a.teardown) and not vm_present(name):
            cmd_network_down(argparse.Namespace(yes=True))
        print("Wait or fetch failed; bucket/account were preserved so the output is recoverable.", file=sys.stderr)
        raise
    if a.cleanup_network or a.teardown:
        cmd_network_down(argparse.Namespace(yes=True))
    if a.teardown:
        cmd_teardown(argparse.Namespace(yes=True, keep_network=True))
    cmd_audit(argparse.Namespace())


def cmd_wait(a):
    require_cloud_config()
    wait_and_finish(a.run, a)


def cmd_status(a):
    require_cloud_config()
    g = gcloud()
    run([g, "compute", "instances", "list", f"--project={PROJECT}", "--filter=labels.nerva=1",
         "--format=table(name,zone.basename(),status,creationTimestamp)"], check=False)
    if a.run:
        log = run([g, "storage", "cat", f"gs://{BUCKET}/runs/{a.run}/out/vm.log"], check=False, capture=True)
        print("\n".join(log.splitlines()[-a.lines:]) if log else "(no log synced yet; the VM syncs every 10 min)")


def cmd_fetch(a):
    require_cloud_config()
    dest = REPO / "experiments" / "cloud_runs" / a.run
    dest.mkdir(parents=True, exist_ok=True)
    run([gcloud(), "storage", "rsync", "--recursive", f"gs://{BUCKET}/runs/{a.run}/out", str(dest)])
    print(f"fetched to {dest}")


def cmd_kill(a):
    require_cloud_config()
    name = instance_name(a.run)
    zone = a.zone or instance_zone(name)
    if not zone:
        print(f"VM {name} is already absent.")
        return
    print(f"Delete VM {name} in {zone} now.")
    confirm(a.yes, "delete it")
    run([gcloud(), "compute", "instances", "delete", name, f"--project={PROJECT}", f"--zone={zone}", "--quiet"])


def cmd_audit(a):
    """Read-only inventory of the resource classes created by this runner."""
    require_cloud_config()
    g = gcloud()
    checks = [
        ("INSTANCES", [g, "compute", "instances", "list", f"--project={PROJECT}",
                       "--format=value(name,zone.basename(),status)"]),
        ("DISKS", [g, "compute", "disks", "list", f"--project={PROJECT}",
                   "--format=value(name,zone.basename(),status,sizeGb)"]),
        ("ADDRESSES", [g, "compute", "addresses", "list", f"--project={PROJECT}",
                       "--format=value(name,region.basename(),address,status)"]),
        ("FORWARDING_RULES", [g, "compute", "forwarding-rules", "list", f"--project={PROJECT}",
                              "--format=value(name,region.basename(),IPAddress)"]),
        ("ROUTERS", [g, "compute", "routers", "list", f"--project={PROJECT}",
                     "--format=value(name,region.basename())"]),
        ("SNAPSHOTS", [g, "compute", "snapshots", "list", f"--project={PROJECT}",
                       "--format=value(name,status)"]),
        ("BUCKETS", [g, "storage", "buckets", "list", f"--project={PROJECT}",
                     "--format=value(name)"]),
        ("RUNNER_SERVICE_ACCOUNT", [g, "iam", "service-accounts", "list", f"--project={PROJECT}",
                                    f"--filter=email={SERVICE_ACCOUNT}", "--format=value(email)"]),
    ]
    def capture(command: list[str]) -> str:
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            detail = (result.stderr or result.stdout or "unknown gcloud error").strip().splitlines()[-1]
            return f"ERROR: {detail}"
        return (result.stdout or "").strip()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(checks)) as executor:
        outputs = list(executor.map(lambda item: capture(item[1]), checks))
    found = False
    had_error = False
    for (label, _), output in zip(checks, outputs):
        print(f"{label}: {output if output else 'none'}")
        had_error = had_error or output.startswith("ERROR:")
        found = found or bool(output) and not output.startswith("ERROR:")
    if not found and not had_error:
        print("Audit result: no runner-related compute, network, snapshot, storage, or identity resources remain.")


def cmd_teardown(a):
    """Remove the dedicated ephemeral resources after outputs are safely local."""
    require_cloud_config()
    g = gcloud()
    instances = run([
        g, "compute", "instances", "list", f"--project={PROJECT}", "--filter=labels.nerva=1",
        "--format=value(name)",
    ], capture=True)
    if instances:
        sys.exit("NERVA VM(s) are still present; teardown refused:\n" + instances)
    print(f"Teardown plan: remove bucket gs://{BUCKET}, runner account {SERVICE_ACCOUNT}, and temporary network.")
    confirm(a.yes, "delete the dedicated NERVA cloud resources")
    if not getattr(a, "keep_network", False):
        cmd_network_down(argparse.Namespace(yes=True))
    if gcloud_succeeds([g, "storage", "buckets", "describe", f"gs://{BUCKET}", f"--project={PROJECT}"]):
        run([g, "storage", "rm", "--recursive", f"gs://{BUCKET}/**"], check=False)
        run([g, "storage", "buckets", "delete", f"gs://{BUCKET}", "--quiet"])
    if gcloud_succeeds([g, "iam", "service-accounts", "describe", SERVICE_ACCOUNT,
                        f"--project={PROJECT}"]):
        run([g, "projects", "remove-iam-policy-binding", PROJECT,
             f"--member=serviceAccount:{SERVICE_ACCOUNT}", "--role=roles/compute.instanceAdmin.v1",
             "--condition=None", "--quiet"], check=False)
        run([g, "iam", "service-accounts", "delete", SERVICE_ACCOUNT,
             f"--project={PROJECT}", "--quiet"])
    print("Dedicated NERVA cloud resources removed.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup"); s.add_argument("--yes", action="store_true"); s.set_defaults(f=cmd_setup)
    s = sub.add_parser("network-up"); s.add_argument("--yes", action="store_true")
    s.set_defaults(f=cmd_network_up)
    s = sub.add_parser("network-down"); s.add_argument("--yes", action="store_true")
    s.set_defaults(f=cmd_network_down)
    s = sub.add_parser("launch")
    s.add_argument("--job", required=True)
    s.add_argument("--hw", choices=sorted(MACHINES), default="l4")
    z = s.add_mutually_exclusive_group()
    z.add_argument("--zone", help="one exact zone; disables automatic fallback")
    z.add_argument("--zones", help="comma-separated fallback order (default: configured zone, then region a/b/c)")
    d = s.add_mutually_exclusive_group()
    d.add_argument("--max-hours", type=int, default=2)
    d.add_argument("--max-minutes", type=int)
    s.add_argument("--spot", action="store_true")
    inputs = s.add_mutually_exclusive_group()
    inputs.add_argument("--input-run", help="R1 cloud run id whose fitted references should be used")
    inputs.add_argument("--input-dir", help="local directory containing styles.json and fitted .pkl files")
    s.add_argument("--wait", action="store_true", help="wait for deletion and fetch results")
    s.add_argument("--cleanup-network", action="store_true",
                   help="with --wait, remove temporary NAT/router after completion")
    s.add_argument("--teardown", action="store_true",
                   help="with --wait, also delete the dedicated bucket and runner account")
    s.add_argument("--poll-seconds", type=int, default=30, help=argparse.SUPPRESS)
    s.add_argument("--yes", action="store_true")
    s.set_defaults(f=cmd_launch)
    s = sub.add_parser("wait", help="resume --wait for a launched run")
    s.add_argument("run"); s.add_argument("--cleanup-network", action="store_true")
    s.add_argument("--teardown", action="store_true")
    s.add_argument("--poll-seconds", type=int, default=30, help=argparse.SUPPRESS)
    s.set_defaults(f=cmd_wait)
    s = sub.add_parser("status"); s.add_argument("run", nargs="?"); s.add_argument("--lines", type=int, default=40)
    s.set_defaults(f=cmd_status)
    s = sub.add_parser("fetch"); s.add_argument("run"); s.set_defaults(f=cmd_fetch)
    s = sub.add_parser("kill"); s.add_argument("run"); s.add_argument("--zone")
    s.add_argument("--yes", action="store_true"); s.set_defaults(f=cmd_kill)
    s = sub.add_parser("audit"); s.set_defaults(f=cmd_audit)
    s = sub.add_parser("teardown"); s.add_argument("--yes", action="store_true")
    s.set_defaults(f=cmd_teardown, keep_network=False)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
