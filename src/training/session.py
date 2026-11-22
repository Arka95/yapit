"""Recording workflow for a profile's independent fine-tuning collection."""

from __future__ import annotations

from yapit.common_utils.metrics import timed
from yapit.enrollment.artifacts import cleanup_session
from yapit.enrollment.profiles import load_voice
from yapit.enrollment.sample_scripts import Reading
from yapit.enrollment.session import RecordingSession
from yapit.models import Sample
from yapit.storage import DataStore
from yapit.training.dataset import (
    MIN_TRAINING_SAMPLES,
    TRAINING_MAX_SECONDS,
    load_training_samples,
    save_training_samples,
)
from yapit.training.sample_scripts import load_readings


class TrainingSession(RecordingSession):
    """Collect training passages without changing the enrolled reference or checkpoint."""

    training = True
    min_seconds = 3
    max_seconds = TRAINING_MAX_SECONDS
    minimum_samples = MIN_TRAINING_SAMPLES

    def __init__(self, store: DataStore, alias: str):
        if not store.voice_path(alias).exists():
            raise ValueError("Enroll this voice first, then open its Training tab.")
        load_voice(store, alias)
        self._existing = load_training_samples(store, alias)
        super().__init__(store, alias)

    def load_readings(self) -> list[Reading]:
        return load_readings()

    def saved_samples(self) -> list[Sample]:
        return self._existing

    @property
    def samples(self) -> list[Sample]:
        combined = {sample.text: sample for sample in self._existing}
        combined.update({sample.text: sample for sample in super().samples})
        return list(combined.values())

    @timed
    def finish(self) -> list[Sample]:
        self._existing = save_training_samples(self.store, self.alias, self.samples)
        available = {sample.text: sample for sample in self._existing}
        self.draft.samples = [available.get(reading.text) for reading in self.draft.readings]
        self.completed()
        cleanup_session(self.store, self.alias, training=True)
        return self._existing
