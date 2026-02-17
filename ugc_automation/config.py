"""
UGC Automation Configuration
----------------------------
Copy this file to config_local.py and fill in your API keys.
Never commit config_local.py to git.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── Anthropic (Claude) ──────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # ── TikTok Content Posting API ─────────────────────────────────────────
    # Get from https://developers.tiktok.com/
    TIKTOK_CLIENT_KEY: str = os.getenv("TIKTOK_CLIENT_KEY", "")
    TIKTOK_CLIENT_SECRET: str = os.getenv("TIKTOK_CLIENT_SECRET", "")
    TIKTOK_ACCESS_TOKEN: str = os.getenv("TIKTOK_ACCESS_TOKEN", "")

    # ── Instagram / Facebook Graph API ─────────────────────────────────────
    # Get from https://developers.facebook.com/
    INSTAGRAM_ACCESS_TOKEN: str = os.getenv("INSTAGRAM_ACCESS_TOKEN", "")
    INSTAGRAM_ACCOUNT_ID: str = os.getenv("INSTAGRAM_ACCOUNT_ID", "")

    # ── ElevenLabs (optional – higher quality TTS) ─────────────────────────
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
    ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "Rachel")  # or any voice ID

    # ── Pexels (free stock video) ──────────────────────────────────────────
    PEXELS_API_KEY: str = os.getenv("PEXELS_API_KEY", "")

    # ── Amazon Associates (optional – for affiliate links) ─────────────────
    AMAZON_AFFILIATE_TAG: str = os.getenv("AMAZON_AFFILIATE_TAG", "")

    # ── Niche & content settings ───────────────────────────────────────────
    # Which product niches to target. The researcher cycles through these.
    NICHES: List[str] = field(default_factory=lambda: [
        "home organization",
        "kitchen gadgets",
        "beauty tools",
        "fitness equipment",
        "tech accessories",
    ])

    # How many products to research per run
    PRODUCTS_PER_RUN: int = 3

    # Video dimensions (TikTok / IG Reels are 9:16)
    VIDEO_WIDTH: int = 1080
    VIDEO_HEIGHT: int = 1920

    # Max video duration in seconds
    MAX_VIDEO_DURATION: int = 45

    # ── Scheduling ─────────────────────────────────────────────────────────
    # Cron-style: post at 8 AM, 12 PM, 6 PM every day
    POST_TIMES: List[str] = field(default_factory=lambda: ["08:00", "12:00", "18:00"])

    # Platforms to post to: "tiktok", "instagram", or both
    PLATFORMS: List[str] = field(default_factory=lambda: ["tiktok", "instagram"])

    # ── Output paths ───────────────────────────────────────────────────────
    OUTPUT_DIR: str = os.path.join(os.path.dirname(__file__), "output")
    SCRIPTS_DIR: str = os.path.join(OUTPUT_DIR, "scripts")
    VIDEOS_DIR: str = os.path.join(OUTPUT_DIR, "videos")
    LOGS_DIR: str = os.path.join(OUTPUT_DIR, "logs")


# Singleton instance used throughout the app
config = Config()
