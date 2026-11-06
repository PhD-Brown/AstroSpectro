"""Guard the offline test environment and shared quality-tool versions."""

from importlib.metadata import version
from pathlib import Path
import socket

import pytest
from pytest_socket import SocketBlockedError
import yaml


@pytest.mark.unit
@pytest.mark.parametrize("tool", ["black", "ruff"])
def test_quality_tool_matches_pre_commit_revision(tool):
    configuration = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / ".pre-commit-config.yaml").read_text(
            encoding="utf-8"
        )
    )
    revisions = [
        repository["rev"].removeprefix("v")
        for repository in configuration["repos"]
        if any(hook["id"] == tool for hook in repository["hooks"])
    ]
    assert revisions == [version(tool)]


@pytest.mark.unit
def test_default_suite_blocks_network_sockets():
    with pytest.raises(SocketBlockedError):
        socket.socket()
