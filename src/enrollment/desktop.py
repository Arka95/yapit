from __future__ import annotations

import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from dotenv import load_dotenv

from yapit.enrollment import (
    MIN_SAMPLES,
    EnrollmentSession,
    MicrophoneRecorder,
    microphone_devices,
    profile_names,
)
from yapit.media import decode_audio
from yapit.storage import DataStore
from yapit.voices import audio_quality


class EnrollmentApp:
    def __init__(self, root: tk.Tk, store: DataStore):
        self.root = root
        self.store = store
        self.session: EnrollmentSession | None = None
        self.recorder = MicrophoneRecorder()
        self.take: Path | None = None
        self.take_ready = False
        self.recording = False
        self.busy = False
        self.started = 0.0
        self.events = queue.Queue()
        self.alias = tk.StringVar()
        self.device = tk.StringVar()
        self.confirmed = tk.BooleanVar(value=False)
        self.normalize = tk.BooleanVar(value=True)
        self.heading = tk.StringVar(value="Create or select a voice profile")
        self.recording_hint = tk.StringVar(
            value="Read naturally for about 8-15 seconds; recording limit: 20 seconds."
        )
        self.status = tk.StringVar(value="Finding microphones...")
        root.title("Yapit - Voice Enrollment")
        root.geometry("860x800")
        root.minsize(760, 760)
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 19, "bold"))
        style.configure("TButton", padding=(12, 8))
        frame = ttk.Frame(root, padding=24)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Your voice, one passage at a time", style="Title.TLabel").pack(
            anchor="w",
            pady=(0, 6),
        )
        ttk.Label(
            frame, text="Local recording only. No uploads, API keys, or model downloads."
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
        self.open_button = ttk.Button(frame, text="Create / open profile", command=self.open_alias)
        self.open_button.pack(anchor="w")
        self.existing = ttk.Label(frame)
        self.existing.pack(anchor="w", pady=(6, 12))
        ttk.Separator(frame).pack(fill="x", pady=(0, 12))
        self.reading_picker = ttk.Combobox(frame, state="readonly")
        self.reading_picker.pack(fill="x", pady=(0, 8))
        self.reading_picker.bind("<<ComboboxSelected>>", self.select_reading)
        ttk.Label(frame, textvariable=self.heading, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.passage = tk.Text(
            frame,
            height=4,
            wrap="word",
            font=("Segoe UI", 17),
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
            "Create or select a profile above, then click Create / open profile.\n\n"
            "Your first reading passage will appear here.",
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
        self.save_button = ttk.Button(controls, text="Save & Next", command=self.save_next)
        for button in (self.record_button, self.stop_button, self.play_button, self.save_button):
            button.pack(side="left", padx=(0, 8))
        self.confirm_check = ttk.Checkbutton(
            frame,
            text="Confirm before saving: I listened back and read the displayed text exactly.",
            variable=self.confirmed,
            command=self.refresh_controls,
        )
        self.confirm_check.pack(anchor="w", pady=(12, 4))
        self.normalize_check = ttk.Checkbutton(
            frame,
            text="Normalize volume and trim outer silence (raw takes kept until Finish).",
            variable=self.normalize,
        )
        self.normalize_check.pack(anchor="w")
        ttk.Label(frame, textvariable=self.status, wraplength=780, justify="left").pack(
            anchor="w",
            fill="x",
            pady=(10, 10),
        )
        self.finish_button = ttk.Button(
            frame,
            text="Finish enrollment",
            command=self.finish_enrollment,
        )
        self.finish_button.pack(anchor="e")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.show_existing()
        self.root.after(100, self.drain_events)
        self.work(microphone_devices, self.devices_loaded)

    def show_existing(self) -> None:
        names = profile_names(self.store)
        self.alias_entry.configure(values=names)
        self.existing.configure(text="Profiles: " + (", ".join(names) or "none yet"))

    def refresh_controls(self) -> None:
        idle = not self.busy and not self.recording
        active = self.session is not None and self.session.index < len(self.session.draft.readings)
        for button, enabled in [
            (self.open_button, idle),
            (self.record_button, idle and active and self.recorder.ready),
            (self.stop_button, self.recording and not self.busy),
            (self.play_button, idle and self.take_ready),
            (self.save_button, idle and active and self.take_ready),
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
                    messagebox.showerror("Yapit", str(error), parent=self.root)
                else:
                    callback(value)
                if not self.root.winfo_exists():
                    return
                self.refresh_controls()
        except queue.Empty:
            pass
        self.root.after(100, self.drain_events)

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
            f"Editing '{session.alias}'. Save at least {MIN_SAMPLES} samples. "
            "Choose a saved passage to replace its audio."
        )
        self.refresh_controls()
        self.prepare_microphone()

    def prepare_microphone(self, event=None) -> None:
        if self.session is None or not self.device.get() or self.busy or self.recording:
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
        if index == total:
            title = f"{session.alias}: all {total} passages saved"
            text = "Finish enrollment, or choose a passage above to record a replacement."
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
        if not self.recorder.ready or captured >= 20 or (captured == 0 and elapsed >= 20):
            self.stop_recording()
            return
        if captured > 0:
            self.record_button.configure(text="Recording")
            self.recording_hint.set("Recording - speak now. Press Stop when finished.")
            self.status.set(f"Recorded audio: {captured:.1f}s / 20s.")
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
            return audio_quality(decode_audio(path, max_seconds=20))

        self.work(stop_and_check, self.recording_stopped)

    def recording_stopped(self, quality: dict) -> None:
        self.take_ready = True
        self.recording_hint.set("Play take, check the confirmation below, then Save & Next.")
        note = " ".join(quality["reasons"]) or "Raw signal checks passed."
        self.status.set(
            f"Recorded {quality['seconds']:.1f}s. {note} "
            "Play it back, then check 'Confirm before saving' and click Save & Next. "
            "Prepared audio is checked on save."
        )

    def save_next(self) -> None:
        if self.session is None or self.take is None:
            return
        if not self.take_ready or self.busy or self.recording:
            return
        if not self.confirmed.get():
            self.status.set(
                "To save: play the take, then check 'Confirm before saving' below the buttons. "
                "Save & Next needs confirmation that you read the displayed text exactly."
            )
            self.confirm_check.focus_set()
            return
        self.stop_playback()
        session, take = self.session, self.take
        confirmed, normalize = self.confirmed.get(), self.normalize.get()
        self.status.set("Checking and saving this passage...")
        self.work(
            lambda: session.save_take(take, confirmed, normalize),
            self.take_saved,
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
                "You can finish enrollment or continue adding samples."
                if self.session.ready
                else f"Save at least {MIN_SAMPLES} samples before finishing."
            )
        )

    def finish_enrollment(self) -> None:
        if self.session is not None:
            self.stop_playback()
            self.status.set("Curating samples and creating the voice alias...")
            session = self.session

            def finish():
                self.recorder.close()
                return session.finish()

            self.work(finish, self.enrollment_finished)

    def enrollment_finished(self, voice) -> None:
        self.show_existing()
        self.recording_hint.set("Enrollment completed; microphone released.")
        self.status.set(
            f"Enrolled '{voice.name}' with {len(voice.samples)} samples. "
            f"Temporary takes and session files removed. "
            f"Files: {self.store.voice_path(voice.name).parent}"
        )
        messagebox.showinfo(
            "Voice enrolled",
            f"'{voice.name}' is ready for reference-based narration.\n"
            "This does not fine-tune model weights; listen to generated speech before publishing.",
            parent=self.root,
        )

    def close(self) -> None:
        if self.busy:
            messagebox.showinfo(
                "Please wait", "Let the current save or stop operation finish.", parent=self.root
            )
            return
        if self.recording:
            if not messagebox.askyesno(
                "Stop recording?",
                "Stop the microphone and close? Previously saved passages remain.",
                parent=self.root,
            ):
                return
            self.recording = False
        self.stop_playback()
        self.work(self.recorder.close, lambda _: self.root.destroy())


def main() -> None:
    if os.name != "nt":
        raise RuntimeError("The native enrollment app currently supports Windows only.")
    load_dotenv()
    root = tk.Tk()
    EnrollmentApp(root, DataStore(Path("data")))
    root.mainloop()


if __name__ == "__main__":
    main()
