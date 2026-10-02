import os

import pytest


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    """Only let global rank zero emit pytest's terminal report."""
    if int(os.environ.get("RANK", "0")) == 0:
        return

    terminal_reporter = config.pluginmanager.get_plugin("terminalreporter")
    if terminal_reporter is not None:
        config.pluginmanager.unregister(terminal_reporter)
