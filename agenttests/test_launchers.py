import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform != "win32" or not shutil.which("py"), reason="Windows py launcher")
@pytest.mark.parametrize(
    ("wrapper", "arguments", "backend", "expected"),
    [
        ("setup-mode.cmd", ["--help"], "setup-mode.py", ["--help"]),
        ("yapit-mode.cmd", ["cpu", "--help"], "run-mode.py", ["cpu", "--help"]),
        ("Start Enrollment.cmd", [], "run-mode.py", ["default", "enroll-app"]),
        ("Start Enrollment.cmd", ["gpu"], "run-mode.py", ["gpu", "enroll-app"]),
    ],
)
def test_shell_launchers_resolve_moved_backends_and_preserve_exit_codes(
    tmp_path, wrapper, arguments, backend, expected
):
    scripts = tmp_path / "project with spaces" / "scripts"
    scripts.mkdir(parents=True)
    original = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("setup-mode.cmd", "yapit-mode.cmd", "Start Enrollment.cmd"):
        shutil.copyfile(original / name, scripts / name)
    stub = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "print(json.dumps({'args': sys.argv[1:], 'backend': Path(__file__).name}))\n"
        "raise SystemExit(7)\n"
    )
    for name in ("setup-mode.py", "run-mode.py"):
        (scripts / name).write_text(stub)
    result = subprocess.run(
        f'call "{scripts / wrapper}" {subprocess.list2cmdline(arguments)}',
        shell=True,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 7, result.stderr
    assert json.loads(result.stdout) == {"args": expected, "backend": backend}
