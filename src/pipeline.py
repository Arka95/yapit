from __future__ import annotations

import traceback
from pathlib import Path

from yapit.common_utils.fileutils import save_json
from yapit.common_utils.mediautils import media_frames
from yapit.common_utils.metrics import timed
from yapit.constants import DEFAULT_PROFILE, VIDEO_DURATION_BUFFER
from yapit.editor import AudioMix, render_guide
from yapit.editor.mixing import video_seconds
from yapit.enrollment.styles import select_voice
from yapit.models import Brief, DeliveryStyle, Script
from yapit.musicgen.models import MusicPlan
from yapit.musicgen.planner import plan_music
from yapit.storage import Workspace
from yapit.yapper import SpeechBackend
from yapit.yapper.generation import render_audio
from yapit.yapper.script_planner import build_prompt, generate_script


class Pipeline:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def prompt(
        self,
        brief: Brief,
        media: Path | None = None,
        profile: str = DEFAULT_PROFILE,
        *,
        render_video: bool = True,
        end_buffer: float = 0.75,
    ) -> dict:
        if not brief.topic and media is None:
            raise ValueError("Provide a topic or a local image/video.")
        job = self.workspace.output(media.stem if media else brief.topic, profile)
        frames = media_frames(media, job.metadata, prefix=job.name) if media else []
        tolerance = brief.seconds * VIDEO_DURATION_BUFFER if render_video else 5
        text = build_prompt(
            brief, duration_tolerance=tolerance, end_buffer=end_buffer if render_video else 0
        )
        job.path("prompt.txt").write_text(text, encoding="utf-8")
        save_json(job.path("brief.json"), brief)
        save_json(job.path("status.json"), {"status": "awaiting_script", "name": job.name})
        return {
            "status": "awaiting_script",
            "job": str(job.metadata),
            "name": job.name,
            "status_file": str(job.path("status.json")),
            "brief": str(job.path("brief.json")),
            "prompt_file": str(job.path("prompt.txt")),
            "prompt": text,
            "frames": [str(frame) for frame in frames],
            "next": "Use this prompt with the frames, then pass the returned Script JSON to reel.",
        }

    @timed
    def reel(
        self,
        brief: Brief,
        voice_name: str = DEFAULT_PROFILE,
        script: Script | None = None,
        media: Path | None = None,
        auto_script: bool = False,
        backend: SpeechBackend | None = None,
        render_video: bool = True,
        style: DeliveryStyle = DeliveryStyle.default,
        source_name: str | None = None,
        music: Path | None = None,
        mix: AudioMix | None = None,
        music_plan: MusicPlan | None = None,
    ) -> dict:
        style = DeliveryStyle(style)
        settings = mix or AudioMix()
        if music is not None and not render_video:
            raise ValueError("Music mixing is video-only; narration output remains separate.")
        if render_video and settings.end_buffer >= brief.seconds:
            raise ValueError("Closing hold must be shorter than the requested video target.")
        if script is None and not auto_script:
            return self.prompt(
                brief,
                media,
                voice_name,
                render_video=render_video,
                end_buffer=settings.end_buffer,
            )
        voice, voice_dir = select_voice(self.workspace.data, voice_name, style)
        brief = brief.model_copy(update={"words_per_minute": voice.words_per_minute})
        source = source_name or (media.stem if media else script.title if script else brief.topic)
        job = self.workspace.output(source, voice_name)
        save_json(job.path("brief.json"), brief)
        save_json(job.path("status.json"), {"status": "running", "name": job.name})
        try:
            transcript_dir = job.transcripts
            cloud_script = script is None and auto_script
            frames = media_frames(media, job.metadata, prefix=job.name) if media else []
            if script is None:
                tolerance = brief.seconds * VIDEO_DURATION_BUFFER if render_video else 5
                script = generate_script(
                    brief,
                    frames,
                    duration_tolerance=tolerance,
                    end_buffer=settings.end_buffer if render_video else 0,
                )
            result, timeline = render_audio(
                script,
                voice,
                voice_dir,
                job,
                seconds=brief.seconds - settings.end_buffer if render_video else brief.seconds,
                style=style,
                backend=backend,
                cloud_script=cloud_script,
                natural_duration_tolerance=brief.seconds * VIDEO_DURATION_BUFFER
                if render_video
                else None,
            )
            if render_video:
                if music is not None:
                    chosen_plan = music_plan or plan_music(script, video_type=brief.kind)
                    save_json(job.path("music-plan.json"), chosen_plan)
                    if not settings.beat_music_levels:
                        settings = AudioMix.model_validate(
                            {
                                **settings.model_dump(),
                                "beat_music_levels": chosen_plan.beat_levels,
                            }
                        )
                save_json(
                    job.path("mix.json"),
                    {
                        "audio": result["audio"],
                        "music": str(music.resolve()) if music else None,
                        "settings": settings.model_dump(mode="json"),
                        "fades_apply_to": "music_only",
                    },
                )
                if abs(video_seconds(timeline, settings) - brief.seconds) > (
                    brief.seconds * VIDEO_DURATION_BUFFER + 1 / timeline.fps
                ):
                    raise ValueError("Video including its closing hold missed the +/-33% buffer.")
                video_duration = render_guide(
                    timeline,
                    Path(result["audio"]),
                    job.metadata,
                    video=job.path("guide.mp4", job.video),
                    captions=job.path("captions.srt", transcript_dir),
                    prefix=job.name,
                    music=music,
                    mix=settings,
                    actual_seconds=result["seconds"],
                )
            result.update(
                {
                    "video": str(job.path("guide.mp4", job.video)) if render_video else None,
                    "captions": str(job.path("captions.srt", transcript_dir))
                    if render_video
                    else None,
                    "video_seconds": video_duration if render_video else None,
                    "music": str(music.resolve()) if music else None,
                    "mix": str(job.path("mix.json")) if render_video else None,
                }
            )
            save_json(job.path("status.json"), result)
            return result
        except Exception as exc:
            save_json(
                job.path("status.json"),
                {
                    "status": "failed",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            job.path("error.log").write_text(traceback.format_exc(), encoding="utf-8")
            raise
