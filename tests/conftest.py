import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="run slow tests (JAX/MJX training-env tests on CPU, several minutes)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow test, run with --runslow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow; use --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
