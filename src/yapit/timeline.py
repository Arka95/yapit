from __future__ import annotations

import re

import numpy as np

from yapit.media import SAMPLE_RATE, split_cut_count
from yapit.models import Cue, Script, Timeline


def build_timeline(
    script: Script,
    durations: list[float],
    seconds: float,
    audio: np.ndarray | None = None,
) -> Timeline:
    if len(durations) != len(script.beats) or any(value <= 0 for value in durations):
        raise ValueError("Each beat needs a positive measured duration.")
    scale = seconds / sum(durations)
    cues = []
    start = 0.0
    for beat_index, (beat, duration) in enumerate(zip(script.beats, durations, strict=True)):
        duration *= scale
        chunks = split_cut_count(duration)
        words = re.findall(r"\S+", beat.text)
        chunks = min(chunks, len(words))
        boundaries = [start]
        for part in range(1, chunks):
            desired = start + duration * part / chunks
            candidate = desired
            if audio is not None:
                # Prefer a quiet 40ms window near the proposed cut, without crossing beat edges.
                centers = np.arange(desired - 0.30, desired + 0.301, 0.02)
                centers = centers[
                    (centers > boundaries[-1] + 0.25) & (centers < start + duration - 0.25)
                ]
                if len(centers):
                    energies = [
                        float(
                            np.mean(
                                audio[
                                    max(0, round((center - 0.02) * SAMPLE_RATE)) : round(
                                        (center + 0.02) * SAMPLE_RATE
                                    )
                                ]
                                ** 2
                            )
                        )
                        for center in centers
                    ]
                    candidate = float(centers[int(np.argmin(energies))])
            boundaries.append(candidate)
        boundaries.append(start + duration)
        for part in range(chunks):
            end = boundaries[part + 1]
            text = " ".join(words[len(words) * part // chunks : len(words) * (part + 1) // chunks])
            cues.append(
                Cue(
                    start=boundaries[part],
                    end=end,
                    text=text,
                    visual=beat.visual,
                    beat=beat_index,
                )
            )
            start = end
    cues[-1].end = seconds
    return Timeline(seconds=seconds, cues=cues)
