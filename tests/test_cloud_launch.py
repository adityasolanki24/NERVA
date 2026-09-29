from argparse import Namespace

from cloud.launch import instance_name, requested_zones


def test_instance_name_sanitizes_job_underscores():
    assert instance_name("b0_baseline-20260929-010203") == "nerva-b0-baseline-20260929-010203"


def test_requested_zones_preserves_fallback_order_without_duplicates():
    args = Namespace(zone=None, zones="us-central1-b,us-central1-a,us-central1-b")
    assert requested_zones(args) == ["us-central1-b", "us-central1-a"]


def test_exact_zone_disables_fallback():
    args = Namespace(zone="us-central1-c", zones=None)
    assert requested_zones(args) == ["us-central1-c"]
