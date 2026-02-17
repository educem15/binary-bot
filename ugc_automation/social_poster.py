"""
Social Media Poster
-------------------
Uploads finished videos to TikTok and Instagram Reels.

TikTok:
  Uses the official TikTok Content Posting API (v2).
  Docs: https://developers.tiktok.com/doc/content-posting-api-get-started

Instagram / Facebook:
  Uses the Instagram Graph API (video reels upload).
  Docs: https://developers.facebook.com/docs/instagram-api/guides/reels

Both flows use a two-step approach:
  1. Initialise upload → get an upload URL
  2. Upload the video binary
  3. Publish
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import requests

from config import config
from script_generator import VideoScript

log = logging.getLogger(__name__)


@dataclass
class PostResult:
    platform: str
    success: bool
    post_id: str = ""
    post_url: str = ""
    error: str = ""


class SocialPoster:
    """Posts videos to TikTok and Instagram."""

    def post(self, video_path: str, script: VideoScript) -> List[PostResult]:
        """Post the video to all configured platforms."""
        results: List[PostResult] = []

        for platform in config.PLATFORMS:
            if platform == "tiktok":
                results.append(self._post_tiktok(video_path, script))
            elif platform == "instagram":
                results.append(self._post_instagram(video_path, script))
            else:
                log.warning("Unknown platform: %s", platform)

        return results

    # ── TikTok ─────────────────────────────────────────────────────────────

    def _post_tiktok(self, video_path: str, script: VideoScript) -> PostResult:
        """Upload video via TikTok Content Posting API."""
        platform = "tiktok"
        if not config.TIKTOK_ACCESS_TOKEN:
            return PostResult(platform=platform, success=False,
                              error="TIKTOK_ACCESS_TOKEN not configured")
        try:
            file_size = os.path.getsize(video_path)

            # Step 1 – Initialise upload
            init_resp = requests.post(
                "https://open.tiktokapis.com/v2/post/publish/video/init/",
                headers={
                    "Authorization": f"Bearer {config.TIKTOK_ACCESS_TOKEN}",
                    "Content-Type": "application/json; charset=UTF-8",
                },
                json={
                    "post_info": {
                        "title": self._build_tiktok_title(script),
                        "privacy_level": "PUBLIC_TO_EVERYONE",
                        "disable_duet": False,
                        "disable_comment": False,
                        "disable_stitch": False,
                        "video_cover_timestamp_ms": 1000,
                    },
                    "source_info": {
                        "source": "FILE_UPLOAD",
                        "video_size": file_size,
                        "chunk_size": file_size,
                        "total_chunk_count": 1,
                    },
                },
                timeout=30,
            )
            init_resp.raise_for_status()
            init_data = init_resp.json().get("data", {})
            publish_id = init_data.get("publish_id", "")
            upload_url = init_data.get("upload_url", "")

            if not upload_url:
                return PostResult(platform=platform, success=False,
                                  error=f"No upload_url in TikTok init response: {init_resp.text}")

            # Step 2 – Upload video binary
            with open(video_path, "rb") as vf:
                upload_resp = requests.put(
                    upload_url,
                    data=vf,
                    headers={
                        "Content-Type": "video/mp4",
                        "Content-Range": f"bytes 0-{file_size - 1}/{file_size}",
                    },
                    timeout=120,
                )
            upload_resp.raise_for_status()

            log.info("TikTok upload complete. publish_id=%s", publish_id)
            return PostResult(
                platform=platform,
                success=True,
                post_id=publish_id,
                post_url=f"https://www.tiktok.com/@me/video/{publish_id}",
            )

        except Exception as exc:
            log.error("TikTok post failed: %s", exc)
            return PostResult(platform=platform, success=False, error=str(exc))

    # ── Instagram ──────────────────────────────────────────────────────────

    def _post_instagram(self, video_path: str, script: VideoScript) -> PostResult:
        """Upload a Reel via Instagram Graph API."""
        platform = "instagram"
        if not config.INSTAGRAM_ACCESS_TOKEN or not config.INSTAGRAM_ACCOUNT_ID:
            return PostResult(platform=platform, success=False,
                              error="INSTAGRAM_ACCESS_TOKEN or INSTAGRAM_ACCOUNT_ID not configured")
        try:
            caption = self._build_instagram_caption(script)

            # Step 1 – Create media container
            #  Note: Instagram requires a publicly accessible video URL.
            #  This implementation uses a resumable upload endpoint.
            container_resp = requests.post(
                f"https://graph.facebook.com/v18.0/{config.INSTAGRAM_ACCOUNT_ID}/media",
                params={"access_token": config.INSTAGRAM_ACCESS_TOKEN},
                json={
                    "media_type": "REELS",
                    "video_url": self._upload_to_temp_host(video_path),
                    "caption": caption,
                    "share_to_feed": True,
                },
                timeout=60,
            )
            container_resp.raise_for_status()
            creation_id = container_resp.json().get("id", "")

            if not creation_id:
                return PostResult(platform=platform, success=False,
                                  error=f"No creation_id returned: {container_resp.text}")

            # Step 2 – Wait for video to be ready
            self._wait_for_instagram_container(creation_id)

            # Step 3 – Publish
            publish_resp = requests.post(
                f"https://graph.facebook.com/v18.0/{config.INSTAGRAM_ACCOUNT_ID}/media_publish",
                params={"access_token": config.INSTAGRAM_ACCESS_TOKEN},
                json={"creation_id": creation_id},
                timeout=30,
            )
            publish_resp.raise_for_status()
            media_id = publish_resp.json().get("id", "")

            log.info("Instagram Reel published. media_id=%s", media_id)
            return PostResult(
                platform=platform,
                success=True,
                post_id=media_id,
                post_url=f"https://www.instagram.com/p/{media_id}/",
            )

        except Exception as exc:
            log.error("Instagram post failed: %s", exc)
            return PostResult(platform=platform, success=False, error=str(exc))

    def _wait_for_instagram_container(self, creation_id: str, max_wait: int = 120):
        """Poll until the container status is FINISHED."""
        deadline = time.time() + max_wait
        while time.time() < deadline:
            resp = requests.get(
                f"https://graph.facebook.com/v18.0/{creation_id}",
                params={
                    "fields": "status_code",
                    "access_token": config.INSTAGRAM_ACCESS_TOKEN,
                },
                timeout=15,
            )
            status = resp.json().get("status_code", "")
            if status == "FINISHED":
                return
            if status == "ERROR":
                raise RuntimeError(f"Instagram container processing failed: {resp.text}")
            log.debug("Instagram container status: %s – waiting…", status)
            time.sleep(5)
        raise TimeoutError("Instagram container did not finish processing in time")

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _build_tiktok_title(script: VideoScript) -> str:
        """TikTok title is limited to 2,200 chars but shows ~150 in feed."""
        hashtag_str = " ".join(f"#{h}" for h in script.hashtags[:10])
        title = f"{script.hook}\n\n{hashtag_str}"
        return title[:2200]

    @staticmethod
    def _build_instagram_caption(script: VideoScript) -> str:
        hashtag_str = " ".join(f"#{h}" for h in script.hashtags[:30])
        return f"{script.caption}\n\n{hashtag_str}"

    @staticmethod
    def _upload_to_temp_host(video_path: str) -> str:
        """
        Instagram requires a PUBLIC video URL.

        Options (you only need one):
          A. Use your own server / S3 bucket and return the URL.
          B. Use a free service like file.io or transfer.sh (short-lived).

        This placeholder uses transfer.sh. Replace with your own solution
        for production use (S3, Cloudflare R2, etc.).
        """
        with open(video_path, "rb") as f:
            resp = requests.put(
                f"https://transfer.sh/{os.path.basename(video_path)}",
                data=f,
                headers={"Max-Downloads": "5", "Max-Days": "1"},
                timeout=120,
            )
        resp.raise_for_status()
        url = resp.text.strip()
        log.info("Video uploaded to temp host: %s", url)
        return url
