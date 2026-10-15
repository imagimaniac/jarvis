import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_sync_gmail_launcher_sets_project_import_path_and_uses_selected_python(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["JARVIS_PYTHON"] = sys.executable
    result = subprocess.run(
        [str(PROJECT_ROOT / "scripts" / "sync_gmail.sh"), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )

    # --help imports sync_gmail.py and its app modules, then exits before OAuth,
    # API access, or opening the database.
    assert result.returncode == 0, result.stderr
    assert "usage: sync_gmail.py" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr
