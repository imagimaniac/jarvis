import os
import subprocess


SCRIPT = os.path.abspath("scripts/python_env.sh")


def _fake_python(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return str(path)


def _selected_python(tmp_path, path, extra_env=None):
    result = subprocess.run(
        ["/bin/bash", "-c", 'source "$1"; jarvis_select_python "$2"; printf "%s" "$PYTHON_BIN"',
         "bash", SCRIPT, str(tmp_path)],
        env={"PATH": path, **(extra_env or {})},
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_python_selection_prefers_project_venv_to_path_python(tmp_path):
    local_python = _fake_python(tmp_path / ".venv" / "bin" / "python")
    fake_bin = tmp_path / "bin"
    _fake_python(fake_bin / "python")
    _fake_python(fake_bin / "python3")

    assert _selected_python(tmp_path, str(fake_bin)) == local_python


def test_python_selection_prefers_active_python_to_python3(tmp_path):
    fake_bin = tmp_path / "bin"
    active_python = _fake_python(fake_bin / "python")
    _fake_python(fake_bin / "python3")

    assert _selected_python(tmp_path, str(fake_bin)) == active_python


def test_python_selection_honors_explicit_jarvis_override(tmp_path):
    fake_bin = tmp_path / "bin"
    _fake_python(fake_bin / "python")
    override = _fake_python(tmp_path / "custom-python")

    assert _selected_python(tmp_path, str(fake_bin), {"JARVIS_PYTHON": override}) == override
