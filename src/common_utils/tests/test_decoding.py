from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from yapit.common_utils.mediautils import decode_audio


@pytest.mark.parametrize("maximum", [None, 190])
def test_decoding_can_preserve_uncapped_audio_without_a_truncation_argument(
    tmp_path, monkeypatch, maximum
):
    media = Mock(return_value=SimpleNamespace(stdout=np.ones(250, dtype="<f4").tobytes()))
    monkeypatch.setattr("yapit.common_utils.mediautils.SAMPLE_RATE", 1)
    monkeypatch.setattr("yapit.common_utils.mediautils.run_media", media)
    source = tmp_path / "narration.mp3"
    if maximum is None:
        assert len(decode_audio(source, max_seconds=None)) == 250
        assert "-t" not in media.call_args.args[0]
    else:
        with pytest.raises(ValueError, match="longer"):
            decode_audio(source, max_seconds=maximum)
        assert "-t" in media.call_args.args[0]
