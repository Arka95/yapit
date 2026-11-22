from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import ImageTk

from yapit.models import Script
from yapit.studio.player import WindowsAudioPlayer
from yapit.studio.workflow import AudioPreview, StudioSession, backend_names
from yapit.yapper.backends import select_backend
from yapit.yapper.model_catalog import LocalModel, local_models


class StudioTabs:
    def __init__(self, app, notebook: ttk.Notebook, session: StudioSession):
        self.app, self.root, self.session = app, app.root, session
        self.player = WindowsAudioPlayer()
        self.playing = False
        self._tick_id = None
        self._image = None
        self._shown_cue = None
        self._script: Script | None = None
        self.voice = tk.StringVar()
        self.backend = tk.StringVar(value=select_backend().backend)
        self.model = tk.StringVar()
        self.model_hint = tk.StringVar()
        self.model_choices: dict[str, LocalModel] = {}
        self.prompt_seconds = tk.StringVar(value="30")
        self.audio_path = tk.StringVar(value="Generate narration or load an audio file.")
        self.audio_output = tk.StringVar()
        self.video_output = tk.StringVar()
        self.clock = tk.StringVar(value="0.00 / 0.00 s")
        self.position = tk.DoubleVar(value=0)
        self.audio_tab = ttk.Frame(notebook, padding=20)
        self.reel_tab = ttk.Frame(notebook, padding=20)
        notebook.add(self.audio_tab, text="3. Audio")
        notebook.add(self.reel_tab, text="4. Reel")
        self._build_audio()
        self._build_reel()
        self.refresh_profiles()

    def _build_audio(self) -> None:
        frame = self.audio_tab
        ttk.Label(frame, text="Create narration", style="Title.TLabel").pack(anchor="w")
        setup = ttk.Frame(frame)
        setup.pack(fill="x", pady=12)
        for column, label in enumerate(("Voice profile", "TTS backend", "Available models")):
            ttk.Label(setup, text=label).grid(row=0, column=column, sticky="w")
            setup.columnconfigure(column, weight=1)
        self.voices = ttk.Combobox(setup, textvariable=self.voice, state="readonly")
        self.backends = ttk.Combobox(
            setup, textvariable=self.backend, values=backend_names(), state="readonly"
        )
        self.models = ttk.Combobox(setup, textvariable=self.model, state="readonly", width=42)
        self.voices.bind("<<ComboboxSelected>>", self.refresh_models)
        self.backends.bind("<<ComboboxSelected>>", self.refresh_models)
        for column, widget in enumerate((self.voices, self.backends, self.models)):
            widget.grid(row=1, column=column, sticky="ew", padx=(0, 10))
        ttk.Label(
            frame,
            textvariable=self.model_hint,
            wraplength=900,
        ).pack(anchor="w")
        ttk.Label(frame, text="Optional AI prompt (cloud opt-in and credentials required)").pack(
            anchor="w", pady=(16, 4)
        )
        self.prompt = tk.Text(frame, height=2, wrap="word")
        self.prompt.pack(fill="x")
        ai = ttk.Frame(frame)
        ai.pack(fill="x", pady=6)
        ttk.Label(ai, text="Script target seconds").pack(side="left")
        self.seconds_entry = ttk.Entry(ai, textvariable=self.prompt_seconds, width=6)
        self.seconds_entry.pack(side="left", padx=8)
        self.ai_button = ttk.Button(ai, text="Ask AI for text", command=self.ask_ai)
        self.ai_button.pack(side="left")
        ttk.Label(
            frame, text="Narration text - type or paste here, then review before generating"
        ).pack(anchor="w", pady=(12, 4))
        self.text = tk.Text(frame, height=9, wrap="word", font=("Segoe UI", 12), undo=True)
        self.text.pack(fill="both", expand=True)
        actions = ttk.Frame(frame)
        actions.pack(fill="x", pady=12)
        self.generate_button = ttk.Button(
            actions, text="Generate audio", command=self.generate_audio
        )
        self.generate_button.pack(side="left")
        self.refresh_button = ttk.Button(
            actions, text="Refresh profiles / models", command=self.refresh_profiles
        )
        self.refresh_button.pack(side="left", padx=8)
        ttk.Label(
            frame,
            text="Audio keeps the reviewed words and natural duration. Models are discovered from "
            "local files; the selected backend must be installed in this app's runtime. "
            "Generation starts only when you click Generate audio.",
            wraplength=820,
        ).pack(anchor="w")
        ttk.Entry(frame, textvariable=self.audio_output, state="readonly").pack(
            fill="x", pady=(10, 0)
        )

    def _build_reel(self) -> None:
        frame = self.reel_tab
        toolbar = ttk.Frame(frame)
        toolbar.pack(fill="x")
        self.load_button = ttk.Button(toolbar, text="Load audio...", command=self.load_audio)
        self.load_button.pack(side="left")
        self.analyze_button = ttk.Button(toolbar, text="Analyze speech beats", command=self.analyze)
        self.analyze_button.pack(side="left", padx=8)
        self.render_button = ttk.Button(
            toolbar, text="Generate video template", command=self.render
        )
        self.render_button.pack(side="left")
        ttk.Label(frame, textvariable=self.audio_path, wraplength=820).pack(anchor="w", pady=8)
        content = ttk.Frame(frame)
        content.pack(fill="both", expand=True)
        self.preview = tk.Canvas(
            content, width=216, height=384, background="#17212b", highlightthickness=0
        )
        self.preview.pack(side="left", padx=(0, 16))
        self.preview.create_text(
            108, 192, text="Video template\npreview", fill="white", justify="center"
        )
        beat_frame = ttk.Frame(content)
        beat_frame.pack(side="left", fill="both", expand=True)
        self.beats = ttk.Treeview(
            beat_frame, columns=("time", "text"), show="headings", selectmode="browse"
        )
        self.beats.heading("time", text="Speech beat / cut")
        self.beats.heading("text", text="Narration / visual")
        self.beats.column("time", width=110, stretch=False)
        self.beats.column("text", width=350)
        scrollbar = ttk.Scrollbar(beat_frame, orient="vertical", command=self.beats.yview)
        self.beats.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.beats.pack(fill="both", expand=True)
        self.beats.bind("<<TreeviewSelect>>", self.select_beat)
        self.waveform = tk.Canvas(frame, height=90, background="#17212b", highlightthickness=0)
        self.waveform.pack(fill="x", pady=(12, 0))
        self.waveform.bind("<Configure>", lambda event: self.draw_waveform())
        self.waveform.bind("<Button-1>", self.seek_waveform)
        self.seekbar = ttk.Scale(frame, variable=self.position, from_=0, to=1)
        self.seekbar.pack(fill="x")
        self.seekbar.bind("<ButtonRelease-1>", lambda event: self.seek(self.position.get()))
        self.seekbar.bind("<KeyRelease>", lambda event: self.seek(self.position.get()))
        controls = ttk.Frame(frame)
        controls.pack(fill="x", pady=6)
        self.play_button = ttk.Button(controls, text="Play / Pause", command=self.play_pause)
        self.play_button.pack(side="left")
        ttk.Label(controls, textvariable=self.clock).pack(side="left", padx=12)
        self.open_button = ttk.Button(controls, text="Open generated MP4", command=self.open_video)
        self.open_button.pack(side="right")
        ttk.Label(
            frame,
            text="Waveform + narration cuts, not music beat detection. Imported audio: up to 180s; "
            "Analyze uses the installed offline ASR model. Generated audio reuses its timing. "
            "MP4 playback opens in your default video player.",
            wraplength=820,
        ).pack(anchor="w")
        ttk.Entry(frame, textvariable=self.video_output, state="readonly").pack(fill="x", pady=6)

    def refresh_profiles(self) -> None:
        from yapit.enrollment.profiles import profile_names

        names = [
            name
            for name in profile_names(self.app.store)
            if self.app.store.voice_path(name).is_file()
        ]
        self.voices.configure(values=names)
        if self.voice.get() not in names:
            self.voice.set(names[0] if names else "")
        self.refresh_models()

    def refresh_models(self, event=None) -> None:
        options = local_models(self.app.store, self.voice.get(), self.backend.get())
        self.model_choices = {option.label: option for option in options}
        self.models.configure(values=list(self.model_choices))
        if self.model.get() not in self.model_choices:
            self.model.set(options[0].label if options else "")
        self.model_hint.set(
            "Select a local model. Trained checkpoints are filtered to the selected voice."
            if options
            else "No compatible local model files found for this backend and voice."
        )
        self.refresh_controls()

    def refresh_controls(self) -> None:
        idle = not self.app.busy and not self.app.recording
        preview = self.session.preview
        for widget, enabled in (
            (self.ai_button, idle),
            (
                self.generate_button,
                idle and bool(self.voice.get()) and self.model.get() in self.model_choices,
            ),
            (self.refresh_button, idle),
            (self.load_button, idle),
            (self.analyze_button, idle and preview is not None and preview.timeline is None),
            (self.render_button, idle and preview is not None and preview.timeline is not None),
            (self.play_button, idle and preview is not None),
            (self.open_button, idle and self.session.video is not None),
            (self.seekbar, idle and preview is not None),
        ):
            widget.configure(state="normal" if enabled else "disabled")
        for widget in (self.voices, self.backends, self.models):
            widget.configure(state="readonly" if idle else "disabled")
        for widget in (self.seconds_entry, self.prompt, self.text):
            widget.configure(state="normal" if idle else "disabled")

    def run(self, operation, callback, status: str) -> None:
        if self.app.busy or self.app.recording:
            return
        self.stop_playback()
        self.app.status.set(status)
        self.app.work(operation, callback)

    def ask_ai(self) -> None:
        from yapit.models import Brief
        from yapit.yapper.script_planner import generate_script

        try:
            prompt = self.prompt.get("1.0", "end").strip()
            brief = Brief(topic=prompt, seconds=float(self.prompt_seconds.get()))
            if not prompt:
                raise ValueError("Enter a prompt for the optional AI draft.")
        except ValueError as exc:
            messagebox.showerror("Script prompt", str(exc), parent=self.root)
            return
        self.run(
            lambda: generate_script(prompt, brief.seconds),
            self.script_ready,
            "Asking AI for a draft. Review the text before generating audio...",
        )

    def script_ready(self, script: Script) -> None:
        self._script = script
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", "\n\n".join(beat.text for beat in script.beats))
        self.app.status.set("Draft ready. Review/edit the narration, then Generate audio.")

    def generate_audio(self) -> None:
        from yapit.yapper.generation import generate_audio

        try:
            text = self.text.get("1.0", "end").strip()
            original = (
                "\n\n".join(beat.text for beat in self._script.beats) if self._script else None
            )
            script = self._script if text == original else Script.from_text(text)
            voice, backend = self.voice.get(), self.backend.get()
            if not voice:
                raise ValueError("Finish enrollment or select an existing voice profile.")
            selected = self.model_choices.get(self.model.get())
            if selected is None:
                raise ValueError("Select an available local model before generating audio.")
        except ValueError as exc:
            messagebox.showerror("Narration", str(exc), parent=self.root)
            return
        self.run(
            lambda: generate_audio(
                script,
                workspace=self.session.workspace,
                voice_name=voice,
                backend_name=backend,
                model=selected.identifier,
            ),
            self.audio_ready,
            "Generating narration with the selected backend...",
        )

    def audio_ready(self, result: dict) -> None:
        self.audio_output.set(result["audio"])
        self.run(
            lambda: self.session.load_audio(Path(result["audio"]), Path(result["timeline"])),
            self.preview_ready,
            "Audio saved. Loading its waveform and measured narration beats...",
        )

    def load_audio(self) -> None:
        from yapit.common_utils.mediautils import SUPPORTED_AUDIO

        path = filedialog.askopenfilename(
            parent=self.root,
            title="Load narration audio",
            filetypes=[("Audio files", " ".join(f"*{ext}" for ext in sorted(SUPPORTED_AUDIO)))],
        )
        if path:
            self.run(
                lambda: self.session.load_audio(Path(path)),
                self.preview_ready,
                "Loading audio waveform...",
            )

    def preview_ready(self, preview: AudioPreview) -> None:
        self.audio_path.set(str(preview.audio))
        self.video_output.set("")
        self.position.set(0)
        self.seekbar.configure(to=preview.seconds)
        self.beats.delete(*self.beats.get_children())
        if preview.timeline:
            for index, cue in enumerate(preview.timeline.cues):
                self.beats.insert(
                    "",
                    "end",
                    iid=str(index),
                    values=(f"{cue.start:.2f} - {cue.end:.2f}s", f"{cue.text} | {cue.visual}"),
                )
        self._shown_cue = None
        self.draw_waveform()
        self.show_position(0)
        self.app.status.set(
            "Audio ready. Play/seek, review the narration cuts, then generate the video template."
            if preview.timeline
            else "Audio ready. Analyze speech beats before generating the video template."
        )

    def analyze(self) -> None:
        self.run(
            self.session.analyze_beats,
            self.preview_ready,
            "Analyzing narration with the installed offline ASR model...",
        )

    def render(self) -> None:
        self.run(
            self.session.generate_video,
            self.video_ready,
            "Rendering the portrait placeholder video template...",
        )

    def video_ready(self, result: dict) -> None:
        self.video_output.set(result["video"])
        self.app.status.set("Video template saved. Open generated MP4 to review it.")

    def open_video(self) -> None:
        if self.session.video is not None:
            try:
                os.startfile(self.session.video)
            except OSError as exc:
                messagebox.showerror("Video player", str(exc), parent=self.root)

    def draw_waveform(self) -> None:
        self.waveform.delete("all")
        preview = self.session.preview
        if preview is None:
            return
        width, height = max(1, self.waveform.winfo_width()), 90
        for index, peak in enumerate(preview.peaks):
            x = index * width / len(preview.peaks)
            amplitude = min(1, peak) * (height / 2 - 4)
            self.waveform.create_line(
                x, height / 2 - amplitude, x, height / 2 + amplitude, fill="#51c5d3"
            )
        if preview.timeline:
            for cue in preview.timeline.cues:
                x = cue.start / preview.seconds * width
                self.waveform.create_line(x, 0, x, height, fill="#efb95e")
        self.waveform.create_line(0, 0, 0, height, fill="white", tags="playhead")
        self._draw_playhead(self.position.get())

    def _draw_playhead(self, seconds: float) -> None:
        preview = self.session.preview
        if preview:
            x = seconds / preview.seconds * self.waveform.winfo_width()
            self.waveform.coords("playhead", x, 0, x, 90)

    def show_position(self, seconds: float) -> None:
        from yapit.editor.templates import create_placeholder

        preview = self.session.preview
        if preview is None:
            return
        self.position.set(seconds)
        self.clock.set(f"{seconds:.2f} / {preview.seconds:.2f} s")
        self._draw_playhead(seconds)
        if preview.timeline is None:
            self.preview.delete("all")
            self.preview.create_text(
                108,
                192,
                text="Analyze speech beats\nto preview the template",
                fill="white",
                justify="center",
            )
            return
        timeline = preview.timeline
        index = next(
            (i for i, cue in enumerate(timeline.cues) if seconds < cue.end),
            len(timeline.cues) - 1,
        )
        if index != self._shown_cue:
            cue = timeline.cues[index]
            thumbnail = timeline.model_copy(update={"width": 216, "height": 384})
            image = create_placeholder(thumbnail, cue, index + 1, alternate=bool(index % 2))
            self._image = ImageTk.PhotoImage(image, master=self.root)
            self.preview.delete("all")
            self.preview.create_image(0, 0, anchor="nw", image=self._image)
            self.beats.see(str(index))
            self._shown_cue = index

    def select_beat(self, event=None) -> None:
        selected = self.beats.selection()
        if selected and self.session.preview and not self.app.busy:
            self.seek(self.session.preview.timeline.cues[int(selected[0])].start)

    def seek_waveform(self, event) -> None:
        preview = self.session.preview
        if preview and not self.app.busy:
            self.seek(event.x / max(1, self.waveform.winfo_width()) * preview.seconds)

    def seek(self, seconds: float) -> None:
        preview = self.session.preview
        if preview is None or self.app.busy or self.app.recording:
            return
        seconds = min(preview.seconds, max(0, seconds))
        if self.playing:
            self._play(seconds)
        self.show_position(seconds)

    def _play(self, seconds: float) -> None:
        try:
            self.player.play(self.session.preview.playback, seconds)
            self.playing = True
        except (OSError, RuntimeError) as exc:
            self.stop_playback()
            messagebox.showerror("Audio player", str(exc), parent=self.root)

    def play_pause(self) -> None:
        if self.playing:
            self.stop_playback()
        elif self.session.preview is not None:
            seconds = self.position.get()
            if seconds >= self.session.preview.seconds - 0.01:
                seconds = 0
            self._play(seconds)
            if self.playing:
                self.playback_tick()

    def playback_tick(self) -> None:
        self._tick_id = None
        if not self.playing:
            return
        try:
            seconds, active = self.player.position()
            self.show_position(seconds)
            self.playing = active
            if active:
                self._tick_id = self.root.after(100, self.playback_tick)
        except (OSError, RuntimeError) as exc:
            self.stop_playback()
            messagebox.showerror("Audio player", str(exc), parent=self.root)

    def stop_playback(self) -> None:
        self.playing = False
        if self._tick_id is not None:
            self.root.after_cancel(self._tick_id)
            self._tick_id = None
        self.player.close()

    def close(self) -> None:
        self.stop_playback()
        self.session.close()
