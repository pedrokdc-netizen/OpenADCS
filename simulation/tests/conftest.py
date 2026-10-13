from pathlib import Path
import subprocess
import sys

import pytest

from openadcs_sim.sensor_processing import NativeProcessing


@pytest.fixture(scope="session")
def native(tmp_path_factory):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path_factory.mktemp("onboard_core")
    completed = subprocess.run(
        [sys.executable, str(root / "tools" / "build_onboard.py"), "--output", str(output)],
        cwd=root, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return NativeProcessing(Path(completed.stdout.strip().splitlines()[-1]))
