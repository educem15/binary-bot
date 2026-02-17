"""
Video Creator
-------------
Turns a VideoScript into a short-form vertical video (1080×1920 / 9:16).

Pipeline:
  1. Generate voiceover audio (ElevenLabs → gTTS fallback)
  2. Fetch relevant stock video clips (Pexels API)
  3. Assemble clips to match voiceover duration (MoviePy)
  4. Overlay animated captions (word-by-word burn-in)
  5. Export MP4 ready for upload

Dependencies:
  pip install moviepy gTTS Pillow requests elevenlabs
"""

import logging
import os
import random
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import requests
from gtts import gTTS
from moviepy.editor import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    TextClip,
    VideoFileClip,
    concatenate_videoclips,
)
from PIL import Image, ImageDraw, ImageFont

from config import config
from script_generator import VideoScript

log = logging.getLogger(__name__)

PEXELS_VIDEO_URL = "https://api.pexels.com/videos/search"
ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


class VideoCreator:
    """Assembles a vertical short-form video from a VideoScript."""

    def __init__(self):
        os.makedirs(config.VIDEOS_DIR, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────

    def create(self, script: VideoScript) -> str:
        """
        Build the video and return the path to the output MP4 file.
        """
        safe_name = "".join(c if c.isalnum() else "_" for c in script.product.name)[:40]
        timestamp = int(time.time())
        output_path = os.path.join(config.VIDEOS_DIR, f"{safe_name}_{timestamp}.mp4")

        with tempfile.TemporaryDirectory() as tmp:
            # Step 1: Generate voiceover
            audio_path = self._generate_voiceover(script.full_voiceover, tmp)

            # Step 2: Get audio duration
            audio_clip = AudioFileClip(audio_path)
            duration = min(audio_clip.duration, config.MAX_VIDEO_DURATION)
            audio_clip.close()

            # Step 3: Fetch and prepare background video clips
            bg_path = self._assemble_background(script.keywords, duration, tmp)

            # Step 4: Compose final video
            self._compose(bg_path, audio_path, script, duration, output_path)

        log.info("Video saved: %s", output_path)
        return output_path

    # ── Step 1: Voiceover ──────────────────────────────────────────────────

    def _generate_voiceover(self, text: str, tmp_dir: str) -> str:
        """Generate MP3 audio. Tries ElevenLabs first, falls back to gTTS."""
        path = os.path.join(tmp_dir, "voiceover.mp3")

        if config.ELEVENLABS_API_KEY:
            try:
                return self._tts_elevenlabs(text, path)
            except Exception as exc:
                log.warning("ElevenLabs TTS failed (%s) – falling back to gTTS", exc)

        return self._tts_gtts(text, path)

    def _tts_elevenlabs(self, text: str, out_path: str) -> str:
        url = ELEVENLABS_TTS_URL.format(voice_id=config.ELEVENLABS_VOICE_ID)
        headers = {
            "xi-api-key": config.ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": "eleven_turbo_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
        log.info("ElevenLabs TTS generated: %s", out_path)
        return out_path

    def _tts_gtts(self, text: str, out_path: str) -> str:
        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(out_path)
        log.info("gTTS voiceover generated: %s", out_path)
        return out_path

    # ── Step 2: Background video ───────────────────────────────────────────

    def _assemble_background(
        self, keywords: List[str], duration: float, tmp_dir: str
    ) -> str:
        """Download stock clips and concatenate to fill `duration` seconds."""
        clips_dir = os.path.join(tmp_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)

        clip_paths = []
        if config.PEXELS_API_KEY:
            clip_paths = self._fetch_pexels_clips(keywords, clips_dir, duration)

        if not clip_paths:
            log.warning("No stock clips found – using colour background")
            return self._make_colour_bg(duration, tmp_dir)

        return self._concat_clips(clip_paths, duration, tmp_dir)

    def _fetch_pexels_clips(
        self, keywords: List[str], clips_dir: str, needed_secs: float
    ) -> List[str]:
        headers = {"Authorization": config.PEXELS_API_KEY}
        paths: List[str] = []
        accumulated = 0.0

        random.shuffle(keywords)
        for kw in keywords:
            if accumulated >= needed_secs:
                break
            try:
                resp = requests.get(
                    PEXELS_VIDEO_URL,
                    headers=headers,
                    params={"query": kw, "orientation": "portrait", "per_page": 5},
                    timeout=15,
                )
                resp.raise_for_status()
                videos = resp.json().get("videos", [])
                random.shuffle(videos)
                for video in videos:
                    if accumulated >= needed_secs:
                        break
                    # Pick the highest-res portrait file
                    files = [
                        f for f in video.get("video_files", [])
                        if f.get("quality") in ("hd", "sd")
                        and f.get("width", 0) < f.get("height", 1)  # portrait
                    ]
                    if not files:
                        continue
                    file_url = files[0]["link"]
                    dest = os.path.join(clips_dir, f"clip_{len(paths)}.mp4")
                    self._download_file(file_url, dest)
                    dur = VideoFileClip(dest).duration
                    paths.append(dest)
                    accumulated += dur
            except Exception as exc:
                log.warning("Pexels fetch error for '%s': %s", kw, exc)

        return paths

    def _concat_clips(
        self, clip_paths: List[str], target_dur: float, tmp_dir: str
    ) -> str:
        clips = []
        total = 0.0
        for p in clip_paths:
            if total >= target_dur:
                break
            clip = VideoFileClip(p).without_audio()
            clip = clip.resize((config.VIDEO_WIDTH, config.VIDEO_HEIGHT))
            remaining = target_dur - total
            if clip.duration > remaining:
                clip = clip.subclip(0, remaining)
            clips.append(clip)
            total += clip.duration

        if not clips:
            return self._make_colour_bg(target_dur, tmp_dir)

        bg = concatenate_videoclips(clips, method="compose")
        out = os.path.join(tmp_dir, "background.mp4")
        bg.write_videofile(out, fps=30, codec="libx264", audio=False, verbose=False, logger=None)
        for c in clips:
            c.close()
        return out

    def _make_colour_bg(self, duration: float, tmp_dir: str) -> str:
        """Fallback: plain dark-gradient background."""
        out = os.path.join(tmp_dir, "background.mp4")
        clip = ColorClip(
            size=(config.VIDEO_WIDTH, config.VIDEO_HEIGHT),
            color=(15, 15, 20),
            duration=duration,
        )
        clip.write_videofile(out, fps=30, codec="libx264", audio=False, verbose=False, logger=None)
        clip.close()
        return out

    # ── Step 3: Compose ────────────────────────────────────────────────────

    def _compose(
        self,
        bg_path: str,
        audio_path: str,
        script: VideoScript,
        duration: float,
        output_path: str,
    ):
        bg = VideoFileClip(bg_path).set_duration(duration)
        audio = AudioFileClip(audio_path).set_duration(duration)

        # Subtitle overlay
        subtitle_clips = self._make_subtitles(script.full_voiceover, duration)

        # Product name watermark (top)
        name_clip = (
            TextClip(
                script.product.name,
                fontsize=42,
                color="white",
                font="DejaVu-Sans-Bold",
                stroke_color="black",
                stroke_width=2,
                method="caption",
                size=(config.VIDEO_WIDTH - 80, None),
            )
            .set_position(("center", 80))
            .set_duration(duration)
        )

        final = CompositeVideoClip(
            [bg, *subtitle_clips, name_clip],
            size=(config.VIDEO_WIDTH, config.VIDEO_HEIGHT),
        ).set_audio(audio)

        final.write_videofile(
            output_path,
            fps=30,
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None,
        )
        bg.close()
        audio.close()
        final.close()

    def _make_subtitles(self, text: str, duration: float) -> List:
        """
        Simple subtitle: split voiceover into ~5-word chunks and display each
        chunk for an equal slice of the video duration.
        """
        words = text.split()
        chunk_size = 5
        chunks = [
            " ".join(words[i: i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]
        if not chunks:
            return []

        time_per_chunk = duration / len(chunks)
        clips = []
        for i, chunk in enumerate(chunks):
            start = i * time_per_chunk
            clip = (
                TextClip(
                    chunk,
                    fontsize=64,
                    color="white",
                    font="DejaVu-Sans-Bold",
                    stroke_color="black",
                    stroke_width=3,
                    method="caption",
                    size=(config.VIDEO_WIDTH - 120, None),
                )
                .set_position(("center", config.VIDEO_HEIGHT * 0.65))
                .set_start(start)
                .set_duration(time_per_chunk)
            )
            clips.append(clip)
        return clips

    # ── Utility ────────────────────────────────────────────────────────────

    @staticmethod
    def _download_file(url: str, dest: str):
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
