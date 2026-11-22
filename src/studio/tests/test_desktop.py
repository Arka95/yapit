from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from yapit.models import Beat, Script
from yapit.studio.desktop import StudioTabs
from yapit.studio.player import WindowsAudioPlayer
from yapit.yapper.model_catalog import LocalModel


@pytest.mark.parametrize("edited", [False, True])
def test_audio_action_keeps_reviewed_words_and_passes_adapter_selection(monkeypatch, edited):
    tab = StudioTabs.__new__(StudioTabs)
    tab._script = Script(title="Draft", beats=[Beat(text="A view.", visual="Panorama")])
    text = "My edited words." if edited else "A view."
    tab.text = Mock()
    tab.text.get.return_value = text
    tab.voice = SimpleNamespace(get=lambda: "myvoice")
    tab.backend = SimpleNamespace(get=lambda: "pocket")
    tab.model = SimpleNamespace(get=lambda: "My installed voice model")
    tab.model_choices = {
        "My installed voice model": LocalModel("My installed voice model", "compatible-model")
    }
    tab.session = SimpleNamespace(workspace=object())
    tab.run = Mock()
    generate = Mock(return_value={"audio": "out.mp3"})
    monkeypatch.setattr("yapit.yapper.generation.generate_audio", generate)
    tab.generate_audio()
    operation = tab.run.call_args.args[0]
    operation()
    args, kwargs = generate.call_args
    assert [beat.text for beat in args[0].beats] == [text]
    if not edited:
        assert args[0] is tab._script
        assert args[0].beats[0].visual == "Panorama"
    assert kwargs == {
        "workspace": tab.session.workspace,
        "voice_name": "myvoice",
        "backend_name": "pocket",
        "model": "compatible-model",
    }


@pytest.mark.parametrize("available", [[], [LocalModel("Pocket English", "english_2026-10")]])
def test_refreshing_models_replaces_stale_backend_selection(monkeypatch, available):
    tab = StudioTabs.__new__(StudioTabs)
    tab.app = SimpleNamespace(store=object())
    tab.voice = SimpleNamespace(get=lambda: "myvoice")
    tab.backend = SimpleNamespace(get=lambda: "pocket")
    values = {"model": "Old Qwen model"}
    tab.model = SimpleNamespace(
        get=lambda: values["model"], set=lambda value: values.update(model=value)
    )
    tab.models, tab.model_hint, tab.refresh_controls = Mock(), Mock(), Mock()
    discover = Mock(return_value=available)
    monkeypatch.setattr("yapit.studio.desktop.local_models", discover)
    tab.refresh_models()
    assert values["model"] == (available[0].label if available else "")
    assert list(tab.model_choices.values()) == available
    discover.assert_called_once_with(tab.app.store, "myvoice", "pocket")


def test_generated_audio_is_loaded_with_its_own_timing():
    tab = StudioTabs.__new__(StudioTabs)
    tab.audio_output = Mock()
    tab.session = Mock()
    tab.run = Mock()
    tab.audio_ready({"audio": "out.mp3", "timeline": "out.timeline.json"})
    tab.run.call_args.args[0]()
    tab.session.load_audio.assert_called_once_with(Path("out.mp3"), Path("out.timeline.json"))
    tab.audio_output.set.assert_called_once_with("out.mp3")


def test_player_seeks_and_reuses_open_media_without_loading_an_sdk(tmp_path):
    player = WindowsAudioPlayer()
    player._command = Mock(return_value="")
    path = tmp_path / "preview.wav"
    player.play(path, 1.25)
    player._command.assert_any_call(f"seek {player.alias} to 1250")
    first_calls = player._command.call_count
    player.play(path, 2)
    assert player._command.call_count == first_calls + 2
    player._command.assert_any_call(f"seek {player.alias} to 2000")
    player._command.side_effect = ["2345", "playing"]
    assert player.position() == (2.345, True)
    player._command.side_effect = None
    player.close()
    assert player.path is None
