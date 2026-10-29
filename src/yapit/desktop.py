from __future__ import annotations

import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from dotenv import load_dotenv

from yapit.enrollment import EnrollmentSession, MicrophoneRecorder, microphone_devices
from yapit.media import decode_audio
from yapit.storage import Workspace
from yapit.voices import audio_quality


class EnrollmentApp:
    def __init__(self, root: tk.Tk, workspace: Workspace):
        self.root = root
        self.workspace = workspace
        self.session: EnrollmentSession | None = None
        self.recorder = MicrophoneRecorder()
        self.take: Path | None = None
        self.recording = False
        self.busy = False
        self.started = 0.0
        self.events = queue.Queue()
        self.alias = tk.StringVar()
        self.device = tk.StringVar()
        self.consent = tk.BooleanVar(value=False)
        self.confirmed = tk.BooleanVar(value=False)
        self.normalize = tk.BooleanVar(value=True)
        self.heading = tk.StringVar(value="Create a voice alias")
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
        ttk.Label(setup, text="Voice alias").grid(row=0, column=0, sticky="w")
        ttk.Label(setup, text="Microphone").grid(row=0, column=1, sticky="w", padx=(16, 0))
        self.alias_entry = ttk.Entry(setup, textvariable=self.alias, width=23)
        self.alias_entry.grid(row=1, column=0, sticky="ew", pady=(4, 8))
        self.devices = ttk.Combobox(setup, textvariable=self.device, state="readonly")
        self.devices.grid(row=1, column=1, sticky="ew", padx=(16, 0), pady=(4, 8))
        setup.columnconfigure(1, weight=1)
        self.consent_check = ttk.Checkbutton(
            frame,
            variable=self.consent,
            text="This is my own voice, and I consent to using these recordings for voice cloning.",
        )
        self.consent_check.pack(anchor="w", pady=(0, 6))
        self.open_button = ttk.Button(frame, text="Open / resume alias", command=self.open_alias)
        self.open_button.pack(anchor="w")
        self.existing = ttk.Label(frame)
        self.existing.pack(anchor="w", pady=(6, 12))
        ttk.Separator(frame).pack(fill="x", pady=(0, 12))
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
            "Choose a new alias above, confirm consent, and click Open / resume alias.\n\n"
            "Your first reading passage will appear here.",
        )
        self.passage.configure(state="disabled")
        self.passage.pack(fill="both", expand=True, pady=(8, 10))
        ttk.Label(
            frame,
            text="Read naturally for about 8-15 seconds. Press Stop when done; 20-second limit.",
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
            text="I listened back and read the displayed text exactly.",
            variable=self.confirmed,
            command=self.refresh_controls,
        )
        self.confirm_check.pack(anchor="w", pady=(12, 4))
        self.normalize_check = ttk.Checkbutton(
            frame,
            text="Normalize volume and trim outer silence (original take is kept).",
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
        names = [
            path.parent.name for path in (self.workspace.root / "voices").glob("*/profile.json")
        ]
        self.existing.configure(
            text="Enrolled aliases: " + (", ".join(sorted(names)) or "none yet")
        )

    def refresh_controls(self) -> None:
        idle = not self.busy and not self.recording
        active = self.session is not None and not self.session.finished
        for button, enabled in [
            (self.open_button, idle),
            (self.record_button, idle and active and bool(self.device.get())),
            (self.stop_button, self.recording and not self.busy),
            (self.play_button, idle and self.take is not None),
            (self.save_button, idle and active and self.take is not None and self.confirmed.get()),
            (
                self.finish_button,
                idle
                and self.session is not None
                and self.session.finished
                and not self.session.draft.completed,
            ),
        ]:
            button.configure(state="normal" if enabled else "disabled")
        self.alias_entry.configure(state="normal" if idle else "disabled")
        self.devices.configure(state="readonly" if idle else "disabled")
        for checkbox in (self.consent_check, self.confirm_check, self.normalize_check):
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
                self.refresh_controls()
        except queue.Empty:
            pass
        self.root.after(100, self.drain_events)

    def devices_loaded(self, devices: list[str]) -> None:
        self.devices.configure(values=devices)
        if devices:
            self.device.set(devices[0])
            self.status.set("Choose a new alias, or enter an unfinished alias to resume it.")
        else:
            self.status.set("No microphone found. Connect a microphone and reopen the app.")

    def open_alias(self) -> None:
        try:
            session = EnrollmentSession(self.workspace, self.alias.get(), self.consent.get())
        except Exception as exc:
            messagebox.showerror("Cannot open alias", str(exc), parent=self.root)
            return
        self.stop_playback()
        self.session = session
        self.take = None
        self.confirmed.set(False)
        self.show_passage()
        self.status.set(f"Editing '{session.alias}'. Saved passages are retained when you close.")
        self.refresh_controls()

    def show_passage(self) -> None:
        session = self.session
        if session is None:
            return
        index = session.index
        total = len(session.draft.readings)
        self.progress.configure(maximum=total, value=index)
        if session.finished:
            title = f"{session.alias}: all {total} passages saved"
            text = "Your recordings and transcripts are saved locally. Click Finish enrollment."
        else:
            reading = session.draft.readings[index]
            title = f"{session.alias}: passage {index + 1} of {total} - {reading.title}"
            text = reading.text
        self.heading.set(title)
        self.passage.configure(state="normal")
        self.passage.delete("1.0", "end")
        self.passage.insert("1.0", text)
        self.passage.configure(state="disabled")

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
        self.stop_playback()
        self.confirmed.set(False)
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
        self.status.set("Starting microphone... wait for 'Recording' before speaking.")
        self.refresh_controls()
        self.root.after(100, self.recording_tick)

    def recording_tick(self) -> None:
        if not self.recording:
            return
        elapsed = time.monotonic() - self.started
        if self.recorder.process.poll() is not None or elapsed >= 20:
            self.stop_recording()
            return
        if self.take and self.take.exists() and self.take.stat().st_size > 44:
            self.status.set(f"Recording: {elapsed:.1f}s / 20s. Speak now; press Stop when done.")
        self.root.after(100, self.recording_tick)

    def stop_recording(self) -> None:
        if not self.recording:
            return
        self.recording = False
        self.status.set("Stopping microphone...")

        def stop_and_check():
            path = self.recorder.stop()
            return audio_quality(decode_audio(path, max_seconds=20))

        self.work(stop_and_check, self.recording_stopped)

    def recording_stopped(self, quality: dict) -> None:
        note = " ".join(quality["reasons"]) or "Raw signal checks passed."
        self.status.set(
            f"Recorded {quality['seconds']:.1f}s. {note} "
            "Play it back, confirm the text, then Save & Next. Prepared audio is checked on save."
        )

    def save_next(self) -> None:
        if self.session is None or self.take is None:
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
        self.confirmed.set(False)
        self.show_passage()
        self.status.set(
            f"Saved {report['quality']['seconds']:.1f}s. "
            + (
                "All passages saved; finish enrollment."
                if self.session.finished
                else "Read the next passage when ready."
            )
        )

    def finish_enrollment(self) -> None:
        if self.session is not None:
            self.status.set("Curating samples and creating the voice alias...")
            self.work(self.session.finish, self.enrollment_finished)

    def enrollment_finished(self, voice) -> None:
        self.show_existing()
        self.status.set(
            f"Enrolled '{voice.name}' with {len(voice.samples)} samples. "
            f"Use this alias for narration. Files: {self.workspace.voice_path(voice.name).parent}"
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
            try:
                self.recorder.stop()
            except Exception:
                pass
        self.stop_playback()
        self.root.destroy()


def main() -> None:
    if os.name != "nt":
        raise RuntimeError("The native enrollment app currently supports Windows only.")
    load_dotenv()
    root = tk.Tk()
    EnrollmentApp(root, Workspace(os.environ.get("YAPIT_WORKSPACE", "workspace")))
    root.mainloop()


if __name__ == "__main__":
    main()
