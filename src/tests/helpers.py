import numpy as np

from yapit.common_utils.mediautils import SAMPLE_RATE


def tone(seconds: float, frequency: float = 180) -> np.ndarray:
    time = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3 * time) ** 2
    return (0.15 * envelope * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
