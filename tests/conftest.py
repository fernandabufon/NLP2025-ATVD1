# tests/conftest.py
from __future__ import annotations

import pytest
import os
import pickle
from pathlib import Path


class Snapshot:
    def __init__(self, snapshot_dir: str = "tests/_snapshots",
                 default_force_update: bool = False,
                 default_test_name: str | None = None):
        self.snapshot_dir = Path(snapshot_dir)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.default_force_update = default_force_update
        self.default_test_name = default_test_name

    def _get_snapshot_path(self, test_name: str) -> Path:
        return self.snapshot_dir / f"{test_name}.pkl"

    def assert_match(self, actual, test_name=None, force_update=False):
        if test_name is None:
            test_name = self.default_test_name
        snapshot_path = self._get_snapshot_path(test_name)
        with open(snapshot_path, "rb") as f:
            expected_data = pickle.load(f)

        if isinstance(actual, dict):
            for key in actual:
                if key not in expected_data:
                    raise AssertionError(f"Key '{key}' not in snapshot")
                assert actual[key] == expected_data[key], (
                    f"Mismatch for key '{key}' in snapshot {test_name}"
                )
        else:
            assert actual == expected_data, f"Data mismatch for {test_name}"

BASE_DIR = Path(__file__).parent  # pega a pasta onde está o conftest.py
SNAPSHOT_DIR = BASE_DIR / "_snapshots"

@pytest.fixture
def snapshot(request):
    return Snapshot(snapshot_dir=SNAPSHOT_DIR, default_test_name=request.node.name)