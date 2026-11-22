"""Training collection, model selection and explicit job submission in the studio."""

from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from yapit.enrollment.desktop import RecordingPanel
from yapit.storage import DataStore
from yapit.training.checkpoints import select_checkpoint
from yapit.training.jobs import run_training
from yapit.training.trainer import TRAINING_MODELS

MODEL_CHOICES = {
    f"Qwen3-TTS {model.split('-12Hz-')[1].removesuffix('-Base')} Base": model
    for model in TRAINING_MODELS
}


class TrainingPanel(RecordingPanel):
    """Keep clip collection and training separate from the profile's reference."""

    def __init__(self, app, frame: ttk.Frame, store: DataStore):
        recording = ttk.Frame(frame)
        recording.pack(side="left", fill="both", expand=True)
        options = ttk.Frame(frame, padding=(20, 0, 0, 0))
        options.pack(side="right", fill="y")
        super().__init__(app, recording, store, training=True)
        self.frame = frame
        self.model = tk.StringVar(value=next(iter(MODEL_CHOICES)))
        self.upstream = tk.StringVar(value=os.environ.get("YAPIT_TRAINING_UPSTREAM", ""))
        self.epochs = tk.StringVar(value="3")
        self.batch_size = tk.StringVar(value="1")
        self.checkpoint = tk.StringVar()
        frame = options
        ttk.Label(frame, text="Fine-tuning", font=("Segoe UI", 13, "bold")).pack(
            anchor="w", pady=(0, 12)
        )
        settings = ttk.Frame(frame)
        settings.pack(fill="x")
        for row, title in enumerate(("Training model", "Epochs", "Batch size")):
            ttk.Label(settings, text=title).grid(row=row * 2, column=0, sticky="w")
        self.models = ttk.Combobox(
            settings, textvariable=self.model, values=list(MODEL_CHOICES), state="readonly"
        )
        self.models.grid(row=1, column=0, sticky="ew", pady=(4, 12))
        self.epochs_entry = ttk.Spinbox(settings, textvariable=self.epochs, from_=1, to=30, width=6)
        self.epochs_entry.grid(row=3, column=0, sticky="w", pady=(4, 12))
        self.batch_entry = ttk.Spinbox(
            settings, textvariable=self.batch_size, from_=1, to=16, width=6
        )
        self.batch_entry.grid(row=5, column=0, sticky="w", pady=(4, 12))
        settings.columnconfigure(0, weight=1)
        checkout = ttk.Frame(frame)
        checkout.pack(fill="x", pady=6)
        ttk.Label(checkout, text="Reviewed Qwen training checkout").pack(anchor="w")
        self.upstream_entry = ttk.Entry(checkout, textvariable=self.upstream)
        self.upstream_entry.pack(fill="x", pady=6)
        self.browse_button = ttk.Button(checkout, text="Browse", command=self.choose_upstream)
        self.browse_button.pack(anchor="w")
        ttk.Label(
            frame,
            text="Training uses Linux/CUDA or the configured WSL2 training runtime. "
            "A missing Base model may download on Train.",
            wraplength=250,
        ).pack(anchor="w", pady=12)
        actions = ttk.Frame(frame)
        actions.pack(fill="x", pady=6)
        self.train_button = ttk.Button(actions, text="Train", command=self.start_training)
        self.train_button.pack(fill="x")
        self.checkpoints = ttk.Combobox(
            actions, textvariable=self.checkpoint, state="readonly", width=28
        )
        self.checkpoints.pack(fill="x", pady=8)
        self.select_button = ttk.Button(
            actions, text="Use reviewed checkpoint", command=self.use_checkpoint
        )
        self.select_button.pack(fill="x")

    def open_alias(self) -> None:
        super().open_alias()
        self.checkpoint.set("")
        self.checkpoints.configure(values=[])

    def choose_upstream(self) -> None:
        directory = filedialog.askdirectory(parent=self.root, title="Qwen3-TTS checkout")
        if directory:
            self.upstream.set(directory)

    def refresh_controls(self) -> None:
        super().refresh_controls()
        idle = not self.app.busy and not self.app.recording
        ready = idle and self.session is not None and self.session.ready and self.take is None
        self.train_button.configure(state="normal" if ready else "disabled")
        self.select_button.configure(
            state="normal" if idle and self.checkpoint.get() else "disabled"
        )
        for widget in (
            self.upstream_entry,
            self.epochs_entry,
            self.batch_entry,
            self.browse_button,
        ):
            widget.configure(state="normal" if idle else "disabled")
        for widget in (self.models, self.checkpoints):
            widget.configure(state="readonly" if idle else "disabled")

    def start_training(self) -> None:
        if self.session is None or self.busy or self.app.recording or self.take is not None:
            return
        try:
            model = MODEL_CHOICES[self.model.get()]
            epochs, batch_size = int(self.epochs.get()), int(self.batch_size.get())
            if not 1 <= epochs <= 30 or not 1 <= batch_size <= 16:
                raise ValueError("Choose 1-30 epochs and batch size 1-16.")
            if not self.upstream.get().strip():
                raise ValueError("Choose the reviewed Qwen training checkout.")
            upstream = Path(self.upstream.get()).resolve()
            if not all(
                (upstream / "finetuning" / filename).is_file()
                for filename in ("prepare_data.py", "sft_12hz.py", "dataset.py")
            ):
                raise ValueError("Choose a Qwen checkout containing the finetuning scripts.")
        except (ValueError, KeyError) as exc:
            messagebox.showerror("Training settings", str(exc), parent=self.root)
            return
        session = self.session
        workspace = self.app.studio.session.workspace
        self.status.set(f"Training '{session.alias}'. The app will report completion or failure.")

        def execute():
            self.recorder.close()
            session.finish()
            return run_training(
                workspace,
                session.alias,
                upstream,
                epochs=epochs,
                batch_size=batch_size,
                model=model,
            )

        self.work(execute, self.training_finished)

    def training_finished(self, result: dict) -> None:
        checkpoints = result["checkpoints"]
        self.checkpoints.configure(values=checkpoints)
        self.checkpoint.set(checkpoints[-1] if checkpoints else "")
        self.checkpoints.bind("<<ComboboxSelected>>", lambda event: self.refresh_controls())
        self.show_passage()
        self.status.set(
            f"Training completed for '{self.session.alias}'. "
            "Review a checkpoint before selecting it for narration."
        )
        self.app.studio.refresh_models()

    def use_checkpoint(self) -> None:
        if self.session is None or not self.checkpoint.get() or self.busy:
            return
        workspace = self.app.studio.session.workspace
        alias, checkpoint = self.session.alias, Path(self.checkpoint.get())
        self.work(
            lambda: select_checkpoint(workspace, alias, checkpoint),
            lambda value: self.checkpoint_selected(alias),
        )

    def checkpoint_selected(self, alias: str) -> None:
        self.app.studio.refresh_models()
        self.status.set(f"Selected the reviewed checkpoint for '{alias}'.")
