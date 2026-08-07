from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config: pytest.Config) -> None:
    """Give each pytest process its own workspace-local temporary directory.

    A unique path avoids Windows ACL conflicts when tests are run from different
    security contexts (for example, a terminal and an isolated IDE runner).
    """
    if config.option.basetemp is None:
        config.option.basetemp = str(
            PROJECT_ROOT / f".pytest_runtime_{uuid4().hex}"
        )
