import os
import subprocess
import sys


def test_extraction_is_model_free_even_with_cuda_asr_settings(tmp_path):
    import numpy as np
    import soundfile as sf

    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(24000, dtype=np.float32), 24000)
    script = """
import importlib.abc
import json
import socket
import sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'transformers', 'qwen_tts', 'faster_whisper',
            'openai', 'huggingface_hub', 'ctranslate2', 'onnxruntime'
        }:
            raise AssertionError('Model dependency accessed: ' + fullname)
sys.meta_path.insert(0, NoModels())
socket.create_connection = lambda *args, **kwargs: (_ for _ in ()).throw(
    AssertionError('Network accessed')
)
from yapit.cli import app
from typer.testing import CliRunner
result = CliRunner().invoke(app, ['extract_audio', sys.argv[1]])
assert result.exit_code == 0, result.output
assert json.loads(result.stdout)['action'] == 'copied'
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(source)],
        env={**os.environ, "YAPIT_ASR_DEVICE": "cuda", "YAPIT_ASR_COMPUTE_TYPE": "float16"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert not (tmp_path / "models").exists()


def test_module_cli_works_without_model_imports_or_network(tmp_path):
    from typer.testing import CliRunner

    from yapit.enrollment.cli import app

    script = """
import importlib.abc
import socket
import sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'transformers', 'qwen_tts', 'faster_whisper', 'openai', 'huggingface_hub'
        }:
            raise AssertionError('Model dependency accessed: ' + fullname)
sys.meta_path.insert(0, NoModels())
socket.create_connection = lambda *args, **kwargs: (_ for _ in ()).throw(
    AssertionError('Network accessed')
)
from yapit.enrollment import EnrollmentSession
from yapit.storage import DataStore
session = EnrollmentSession(DataStore(sys.argv[1]), 'TESTUSER')
assert session.alias == 'testuser'
assert not session.ready
import numpy as np
from yapit.common_utils.mediautils import SAMPLE_RATE, write_wav
for index in range(5):
    path = session.new_take()
    t = np.arange(4 * SAMPLE_RATE) / SAMPLE_RATE
    write_wav(path, (0.15 * np.sin(2 * np.pi * (180 + index * 30) * t)).astype(np.float32))
    session.save_take(path, True, False)
assert len(session.finish().samples) == 5
from yapit.enrollment.cli import app
from typer.testing import CliRunner
result = CliRunner().invoke(app, ['profiles'])
assert result.exit_code == 0, result.output
from yapit.cli import app as main_app
from pathlib import Path
result = CliRunner().invoke(main_app, [
    'voice-enroll', 'testuser', '--manifest', str(session.folder / 'testuser.samples.json')
])
assert result.exit_code == 0, result.output
assert not (Path(sys.argv[2]) / 'enrollment' / 'testuser').exists()
assert not list(session.store.root.rglob('*.log'))
assert not list(session.store.root.rglob('*.raw.*'))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "data"), str(tmp_path / "workspace")],
        env={**os.environ, "YAPIT_WORKSPACE": str(tmp_path / "workspace")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    output = CliRunner().invoke(app, ["profiles"])
    assert output.exit_code == 0
