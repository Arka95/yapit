from __future__ import annotations

import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from dotenv import load_dotenv

from yapit.common_utils.mediautils import decode_audio
from yapit.common_utils.metrics import configure_logging
from yapit.enrollment import (
    EnrollmentSession,
    MicrophoneRecorder,
    microphone_devices,
    profile_names,
)
from yapit.enrollment.profiles import audio_quality
from yapit.enrollment.sample_scripts import ENROLLMENT_MAX_SECONDS
from yapit.enrollment.session import RecordingSession
from yapit.storage import DataStore, Workspace


class RecordingPanel:
    """Record/review controls shared by the enrollment and training tabs."""

    def __init__(self, app, frame: ttk.Frame, store: DataStore, *, training: bool = False):
        self.app, self.root, self.frame = app, app.root, frame
        self.store = store
        self.training = training
        self.session: RecordingSession | None = None
        self.recorder = MicrophoneRecorder(max_seconds=20 if training else ENROLLMENT_MAX_SECONDS)
        self.take: Path | None = None
        self.take_ready = False
        self.recording = False
        self.started = 0.0
        self.alias = tk.StringVar()
        self.device = tk.StringVar()
        self.confirmed = tk.BooleanVar(value=False)
        self.normalize = tk.BooleanVar(value=True)
        self.heading = tk.StringVar(value="Create or select a voice profile")
        self.recording_hint = tk.StringVar(
            value=(
                "Read each training passage naturally; 3-20 seconds per clip."
                if training
                else "Read the whole paragraph naturally in about one minute; 45-90 seconds."
            )
        )
        self.status = app.status
        ttk.Label(
            frame,
            text="Training recordings" if training else "Enroll your voice in one recording",
            style="Title.TLabel",
        ).pack(
            anchor="w",
            pady=(0, 6),
        )
        ttk.Label(
            frame,
            text=(
                "Collect 16 short passages for an existing profile, or reuse its saved clips."
                if training
                else "One phonetically varied paragraph becomes your reusable voice reference."
            ),
        ).pack(
            anchor="w",
            pady=(0, 16),
        )
        setup = ttk.Frame(frame)
        setup.pack(fill="x")
        ttk.Label(setup, text="Voice profile").grid(row=0, column=0, sticky="w")
        ttk.Label(setup, text="Microphone").grid(row=0, column=1, sticky="w", padx=(16, 0))
        self.alias_entry = ttk.Combobox(setup, textvariable=self.alias, width=23)
        self.alias_entry.grid(row=1, column=0, sticky="ew", pady=(4, 8))
        self.devices = ttk.Combobox(setup, textvariable=self.device, state="readonly")
        self.devices.bind("<<ComboboxSelected>>", self.prepare_microphone)
        self.devices.grid(row=1, column=1, sticky="ew", padx=(16, 0), pady=(4, 8))
        setup.columnconfigure(1, weight=1)
        self.open_button = ttk.Button(
            frame,
            text="Load profile" if training else "Create / open profile",
            command=self.open_alias,
        )
        self.open_button.pack(anchor="w")
        self.existing = ttk.Label(frame)
        self.existing.pack(anchor="w", pady=(6, 12))
        ttk.Separator(frame).pack(fill="x", pady=(0, 12))
        self.reading_picker = ttk.Combobox(frame, state="readonly")
        if training:
            self.reading_picker.pack(fill="x", pady=(0, 8))
        self.reading_picker.bind("<<ComboboxSelected>>", self.select_reading)
        ttk.Label(frame, textvariable=self.heading, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.passage = tk.Text(
            frame,
            height=4,
            wrap="word",
            font=("Segoe UI", 12 if training else 15),
            padx=14,
            pady=14,
            background="#ffffff",
            relief="solid",
            borderwidth=1,
            state="disabled",
        )
        self.passage.configure(state="normal")
        self.passage.insert(
            "1.0",
            (
                "Load an enrolled profile to collect or review its training passages."
                if training
                else "Create or open a profile to read its one-minute enrollment paragraph."
            ),
        )
        self.passage.configure(state="disabled")
        self.passage.pack(fill="both", expand=True, pady=(8, 10))
        ttk.Label(
            frame,
            textvariable=self.recording_hint,
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w")
        self.progress = ttk.Progressbar(frame, mode="determinate")
        self.progress.pack(fill="x", pady=(8, 12))
        controls = ttk.Frame(frame)
        controls.pack(fill="x")
        self.record_button = ttk.Button(controls, text="Record", command=self.start_recording)
        self.stop_button = ttk.Button(controls, text="Stop", command=self.stop_recording)
        self.play_button = ttk.Button(controls, text="Play take", command=self.play_take)
        self.save_button = ttk.Button(
            controls,
            text="Save & Next" if training else "Enroll voice",
            command=self.save_next,
        )
        for button in (self.record_button, self.stop_button, self.play_button, self.save_button):
            button.pack(side="left", padx=(0, 8))
        self.confirm_check = ttk.Checkbutton(
            frame,
            text="Confirm before saving: I replayed and read the complete text exactly.",
            variable=self.confirmed,
            command=self.refresh_controls,
        )
        self.confirm_check.pack(anchor="w", pady=(12, 4))
        self.normalize_check = ttk.Checkbutton(
            frame,
            text="Normalize volume and trim outer silence.",
            variable=self.normalize,
        )
        self.normalize_check.pack(anchor="w")
        self.finish_button = ttk.Button(
            frame,
            text="Save training recordings",
            command=self.finish_enrollment,
        )
        if training:
            self.finish_button.pack(anchor="e")
        self.show_existing()

    @property
    def busy(self) -> bool:
        return self.app.busy

    def work(self, operation, callback) -> None:
        self.app.work(operation, callback)

    def show_existing(self) -> None:
        names = profile_names(self.store)
        self.alias_entry.configure(values=names)
        self.existing.configure(text="Profiles: " + (", ".join(names) or "none yet"))

    def refresh_controls(self) -> None:
        idle = not self.busy and not self.recording
        active = self.session is not None and (
            not self.training or self.session.index < len(self.session.draft.readings)
        )
        for button, enabled in [
            (self.open_button, idle),
            (self.record_button, idle and active and self.recorder.ready),
            (self.stop_button, self.recording and not self.busy),
            (self.play_button, idle and self.take_ready),
            (
                self.save_button,
                idle
                and (
                    (active and self.take_ready)
                    or (
                        not self.training
                        and self.session is not None
                        and self.session.ready
                        and not self.session.draft.completed
                    )
                ),
            ),
            (
                self.finish_button,
                idle
                and self.session is not None
                and self.session.ready
                and self.take is None
                and not self.session.draft.completed,
            ),
        ]:
            button.configure(state="normal" if enabled else "disabled")
        self.alias_entry.configure(state="normal" if idle else "disabled")
        self.devices.configure(state="readonly" if idle else "disabled")
        self.reading_picker.configure(state="readonly" if idle and self.session else "disabled")
        for checkbox in (self.confirm_check, self.normalize_check):
            checkbox.configure(state="normal" if idle else "disabled")

    def devices_loaded(self, devices: list[str]) -> None:
        self.devices.configure(values=devices)
        if devices:
            self.device.set(devices[0])
            if self.session:
                self.prepare_microphone()
            else:
                self.status.set("Create or select a profile to prepare the microphone.")
        else:
            self.status.set("No microphone found. Connect a microphone and reopen the app.")

    def open_alias(self) -> None:
        try:
            if self.training:
                from yapit.training.session import TrainingSession

                session = TrainingSession(self.store, self.alias.get())
            else:
                session = EnrollmentSession(self.store, self.alias.get())
        except Exception as exc:
            messagebox.showerror("Cannot open alias", str(exc), parent=self.root)
            return
        self.stop_playback()
        self.session = session
        self.alias.set(session.alias)
        self.show_existing()
        self.reading_picker.configure(
            values=[
                f"{reading.sample_id or f'sample_{index + 1}'} - {reading.title}"
                for index, reading in enumerate(session.draft.readings)
            ]
        )
        self.take = None
        self.take_ready = False
        self.confirmed.set(False)
        self.recording_hint.set("Preparing microphone...")
        self.show_passage()
        self.status.set(
            f"'{session.alias}': {len(session.samples)} training clips available. "
            "Choose a passage to record or replace a training clip."
            if self.training
            else f"'{session.alias}': record the complete paragraph, replay it, "
            "confirm the words, then click Enroll voice."
        )
        self.refresh_controls()
        self.prepare_microphone()

    def prepare_microphone(self, event=None) -> None:
        if (
            self.session is None
            or not self.device.get()
            or self.busy
            or self.recording
            or self.app.notebook.select() != str(self.frame)
        ):
            return
        device, directory = self.device.get(), self.session.scratch
        self.recording_hint.set("Preparing microphone - Record will enable when ready.")
        self.status.set("Microphone stays active between takes; only Record saves audio.")
        self.work(
            lambda: self.recorder.prepare(device, directory),
            self.microphone_prepared,
        )

    def microphone_prepared(self, value=None) -> None:
        self.recording_hint.set("Microphone ready. Record starts immediately.")
        self.status.set("Record to save audio. Stop ends the take; Close releases the microphone.")

    def show_passage(self) -> None:
        session = self.session
        if session is None:
            return
        index = session.index
        total = len(session.draft.readings)
        self.progress.configure(
            maximum=total, value=sum(sample is not None for sample in session.draft.samples)
        )
        if not self.training:
            title = f"{session.alias}: one-minute enrollment paragraph"
            text = session.draft.readings[0].text
            self.record_button.configure(
                text="Record replacement" if session.draft.completed else "Record"
            )
        elif index == total:
            title = f"{session.alias}: all {total} passages saved"
            text = "Save the training collection, or choose a passage to record a replacement."
            self.reading_picker.set("")
        else:
            reading = session.draft.readings[index]
            title = f"{session.alias}: passage {index + 1} of {total} - {reading.title}"
            text = reading.text
            self.reading_picker.current(index)
        self.heading.set(title)
        self.passage.configure(state="normal")
        self.passage.delete("1.0", "end")
        self.passage.insert("1.0", text)
        self.passage.configure(state="disabled")

    def select_reading(self, event=None) -> None:
        if self.session and not self.busy and not self.recording:
            self.stop_playback()
            self.session.select_sample(self.reading_picker.current())
            self.take = None
            self.take_ready = False
            self.confirmed.set(False)
            self.recording_hint.set("Microphone ready. Record starts immediately.")
            self.show_passage()
            self.refresh_controls()
            self.prepare_microphone()

    @staticmethod
    def stop_playback() -> None:
        import winsound

        winsound.PlaySound(None, 0)

    def play_take(self) -> None:
        if self.take:
            import winsound

            winsound.PlaySound(str(self.take), winsound.SND_FILENAME | winsound.SND_ASYNC)

    def start_recording(self) -> None:
        if self.session is None:
            return
        if not self.training and self.session.index == len(self.session.draft.readings):
            self.session.select_sample(0)
        if not self.recorder.ready:
            self.prepare_microphone()
            return
        self.stop_playback()
        self.confirmed.set(False)
        self.take_ready = False
        try:
            self.take = self.session.new_take()
            self.recorder.start(self.device.get(), self.take)
        except Exception as exc:
            self.take = None
            self.status.set(str(exc))
            messagebox.showerror("Cannot record", str(exc), parent=self.root)
            return
        self.started = time.monotonic()
        self.recording = True
        self.record_button.configure(text="Recording")
        self.recording_hint.set("Recording - speak now. Press Stop when finished.")
        self.status.set("Recording. The microphone was already prepared.")
        self.refresh_controls()
        self.root.after(100, self.recording_tick)

    def recording_tick(self) -> None:
        if not self.recording:
            return
        elapsed = time.monotonic() - self.started
        captured = self.recorder.captured_seconds
        limit = self.recorder.max_seconds
        if not self.recorder.ready or captured >= limit or (captured == 0 and elapsed >= limit):
            self.stop_recording()
            return
        if captured > 0:
            self.record_button.configure(text="Recording")
            self.recording_hint.set("Recording - speak now. Press Stop when finished.")
            self.status.set(f"Recorded audio: {captured:.1f}s / {limit}s.")
        self.root.after(100, self.recording_tick)

    def stop_recording(self) -> None:
        if not self.recording:
            return
        self.recording = False
        self.record_button.configure(text="Record")
        self.recording_hint.set("Recording stopped. Preparing playback...")
        self.status.set("Stopping microphone...")

        def stop_and_check():
            path = self.recorder.stop()
            return audio_quality(
                decode_audio(path, max_seconds=self.session.max_seconds),
                min_seconds=self.session.min_seconds,
                max_seconds=self.session.max_seconds,
            )

        self.work(stop_and_check, self.recording_stopped)

    def recording_stopped(self, quality: dict) -> None:
        self.take_ready = True
        action = "Save & Next" if self.training else "Enroll voice"
        self.recording_hint.set(f"Play take, check the confirmation below, then {action}.")
        note = " ".join(quality["reasons"]) or "Raw signal checks passed."
        self.status.set(
            f"Recorded {quality['seconds']:.1f}s. {note} "
            f"Play it back, then check 'Confirm before saving' and click {action}. "
            "Prepared audio is checked on save."
        )

    def save_next(self) -> None:
        if self.session is None or self.busy or self.recording:
            return
        if self.take is None:
            if not self.training and self.session.ready and not self.session.draft.completed:
                session = self.session

                def finish_saved():
                    self.recorder.close()
                    return session.finish()

                self.work(finish_saved, self.enrollment_finished)
            return
        if not self.take_ready:
            return
        if not self.confirmed.get():
            self.status.set(
                "To save: play the take, then check 'Confirm before saving' below the buttons. "
                "Confirm that you read the complete displayed text exactly."
            )
            self.confirm_check.focus_set()
            return
        self.stop_playback()
        session, take = self.session, self.take
        confirmed, normalize = self.confirmed.get(), self.normalize.get()
        self.status.set(
            "Checking and saving this training passage..."
            if self.training
            else "Checking your recording and enrolling the voice..."
        )

        def save():
            report = session.save_take(take, confirmed, normalize)
            if self.training:
                return report
            self.recorder.close()
            return session.finish()

        self.work(
            save,
            self.take_saved if self.training else self.enrollment_finished,
        )

    def take_saved(self, report: dict) -> None:
        self.take = None
        self.take_ready = False
        self.confirmed.set(False)
        self.recording_hint.set("Microphone ready. Record starts immediately.")
        self.show_passage()
        self.status.set(
            f"Saved {report['quality']['seconds']:.1f}s. "
            + (
                "You can save the training collection or continue recording."
                if self.session.ready
                else f"Collect at least {self.session.minimum_samples} training clips."
            )
        )

    def finish_enrollment(self) -> None:
        if self.session is not None:
            self.stop_playback()
            self.status.set("Checking and saving the training collection...")
            session = self.session

            def finish():
                self.recorder.close()
                return session.finish()

            self.work(finish, self.training_saved)

    def training_saved(self, samples) -> None:
        self.take = None
        self.take_ready = False
        self.show_passage()
        self.status.set(f"Saved {len(samples)} training clips for '{self.session.alias}'.")

    def enrollment_finished(self, voice) -> None:
        self.take = None
        self.take_ready = False
        self.confirmed.set(False)
        self.show_passage()
        self.app.refresh_profiles()
        self.app.studio.voice.set(voice.name)
        self.app.studio.refresh_models()
        self.recording_hint.set("Enrollment completed; microphone released.")
        self.status.set(
            f"Enrolled '{voice.name}' with one reference recording. "
            f"Temporary takes and session files removed. "
            f"Files: {self.store.voice_path(voice.name).parent}"
        )
        messagebox.showinfo(
            "Voice enrolled",
            f"'{voice.name}' is ready for reference-based narration.\n"
            "Use the Training tab to collect a fine-tuning dataset.",
            parent=self.root,
        )


class EnrollmentApp:
    """Desktop studio shell with one shared worker and separate recording workflows."""

    def __init__(self, root: tk.Tk, store: DataStore):
        from yapit.studio.desktop import StudioTabs
        from yapit.studio.workflow import StudioSession
        from yapit.training.desktop import TrainingPanel

        self.root, self.store = root, store
        self.busy = False
        self._closed = False
        self.events = queue.Queue()
        self.status = tk.StringVar(value="Finding microphones...")
        root.title("YapIt - Voice, Audio & Reels")
        root.geometry("1080x900")
        root.minsize(1000, 820)
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 19, "bold"))
        style.configure("TButton", padding=(12, 8))
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)
        enrollment_tab = ttk.Frame(self.notebook, padding=24)
        training_tab = ttk.Frame(self.notebook, padding=24)
        self.notebook.add(enrollment_tab, text="1. Enrollment")
        self.notebook.add(training_tab, text="2. Training")
        self.enrollment = RecordingPanel(self, enrollment_tab, store)
        self.training = TrainingPanel(self, training_tab, store)
        self.panels = (self.enrollment, self.training)
        workspace = Workspace(os.environ.get("YAPIT_WORKSPACE", "workspace"))
        workspace.data = store
        self.studio = StudioTabs(self, self.notebook, StudioSession(workspace))
        ttk.Label(root, textvariable=self.status, wraplength=1000, padding=12).pack(fill="x")
        self.notebook.bind("<<NotebookTabChanged>>", self.tab_changed)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self.drain_events)
        self.work(microphone_devices, self.devices_loaded)

    @property
    def recording(self) -> bool:
        return any(panel.recording for panel in self.panels)

    def refresh_profiles(self) -> None:
        for panel in self.panels:
            panel.show_existing()
        self.studio.refresh_profiles()

    def refresh_controls(self) -> None:
        for panel in self.panels:
            panel.refresh_controls()
        self.studio.refresh_controls()

    def devices_loaded(self, devices: list[str]) -> None:
        for panel in self.panels:
            panel.devices_loaded(devices)

    def tab_changed(self, event=None) -> None:
        for panel in self.panels:
            if panel.recording:
                self.notebook.select(panel.frame)
                return
        self.studio.stop_playback()
        RecordingPanel.stop_playback()
        if self.busy:
            return
        selected = self.notebook.select()
        inactive = [panel for panel in self.panels if str(panel.frame) != selected]

        def release():
            for panel in inactive:
                panel.recorder.close()

        def prepare(value):
            for panel in self.panels:
                if str(panel.frame) == self.notebook.select():
                    panel.show_existing()
                    panel.prepare_microphone()

        self.work(release, prepare)

    def work(self, operation, callback) -> None:
        if self.busy:
            return
        self.busy = True
        self.refresh_controls()

        def execute():
            try:
                self.events.put((callback, operation(), None))
            except Exception as exc:
                self.events.put((callback, None, exc))

        threading.Thread(target=execute, daemon=True).start()

    def drain_events(self) -> None:
        try:
            while True:
                callback, value, error = self.events.get_nowait()
                self.busy = False
                if error:
                    self.status.set(str(error))
                    messagebox.showerror("YapIt", str(error), parent=self.root)
                else:
                    try:
                        callback(value)
                    except Exception as exc:
                        self.status.set(str(exc))
                        messagebox.showerror("YapIt", str(exc), parent=self.root)
                if self._closed:
                    return
                self.refresh_controls()
                if not self.busy and any(
                    panel.recorder.ready and str(panel.frame) != self.notebook.select()
                    for panel in self.panels
                ):
                    self.tab_changed()
        except queue.Empty:
            pass
        self.root.after(100, self.drain_events)

    def close(self) -> None:
        if self.busy:
            messagebox.showinfo(
                "Please wait", "Let the current operation finish before closing.", parent=self.root
            )
            return
        if self.recording:
            if not messagebox.askyesno(
                "Stop recording?",
                "Stop the microphone and close? Previously saved passages remain.",
                parent=self.root,
            ):
                return
            for panel in self.panels:
                panel.recording = False
        RecordingPanel.stop_playback()
        self.studio.close()

        def release():
            for panel in self.panels:
                panel.recorder.close()

        self.work(release, self.window_closed)

    def window_closed(self, value=None) -> None:
        self._closed = True
        self.root.destroy()


def main() -> None:
    if os.name != "nt":
        raise RuntimeError("The native enrollment app currently supports Windows only.")
    load_dotenv()
    configure_logging()
    root = tk.Tk()
    EnrollmentApp(root, DataStore())
    root.mainloop()


if __name__ == "__main__":
    main()
