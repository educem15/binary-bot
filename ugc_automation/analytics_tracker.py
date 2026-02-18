"""
Analytics Tracker
-----------------
SQLite-based performance tracker. Records every post, fetches engagement
metrics from TikTok / Instagram APIs, and surfaces insights that feed
back into the script generator so the pipeline improves automatically.

Key features:
  - Tracks every post (platform, niche, product, hook, hashtags)
  - Fetches live metrics (views, likes, shares, comments, saves)
  - Ranks niches and hook styles by performance
  - Surfaces the best posting times per platform
  - Prevents duplicate product usage
  - Generates weekly performance reports
"""

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests

from config import config

log = logging.getLogger(__name__)

DB_PATH = os.path.join(config.OUTPUT_DIR, "analytics.db")

TIKTOK_VIDEO_QUERY_URL = "https://open.tiktokapis.com/v2/video/query/"
INSTAGRAM_MEDIA_URL = "https://graph.facebook.com/v18.0/{media_id}"


@dataclass
class PostRecord:
    platform: str
    post_id: str
    product_name: str
    niche: str
    hook: str
    hashtags: List[str]
    video_path: str
    posted_at: str                  # ISO datetime
    views: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    saves: int = 0
    affiliate_clicks: int = 0
    revenue_usd: float = 0.0
    last_synced_at: str = ""


@dataclass
class NichePerformance:
    niche: str
    total_posts: int
    avg_views: float
    avg_likes: float
    avg_shares: float
    top_hook: str
    best_post_time: str
    total_revenue: float


class AnalyticsTracker:
    """Records, syncs, and analyses post performance."""

    def __init__(self):
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        self._init_db()

    # ── Public API ─────────────────────────────────────────────────────────

    def record_post(self, record: PostRecord):
        """Save a newly published post to the database."""
        with self._db() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO posts (
                    platform, post_id, product_name, niche, hook,
                    hashtags, video_path, posted_at,
                    views, likes, comments, shares, saves,
                    affiliate_clicks, revenue_usd, last_synced_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.platform, record.post_id, record.product_name,
                    record.niche, record.hook,
                    json.dumps(record.hashtags), record.video_path,
                    record.posted_at, record.views, record.likes,
                    record.comments, record.shares, record.saves,
                    record.affiliate_clicks, record.revenue_usd,
                    record.last_synced_at,
                ),
            )
        log.info("Recorded post: [%s] %s (%s)", record.platform, record.product_name, record.post_id)

    def sync_metrics(self):
        """Fetch latest engagement numbers for all posts from each platform API."""
        posts = self._get_unsynced_posts()
        log.info("Syncing metrics for %d posts…", len(posts))

        for row in posts:
            platform, post_id = row["platform"], row["post_id"]
            try:
                if platform == "tiktok":
                    metrics = self._fetch_tiktok_metrics(post_id)
                elif platform == "instagram":
                    metrics = self._fetch_instagram_metrics(post_id)
                else:
                    continue

                if metrics:
                    self._update_metrics(post_id, platform, metrics)
            except Exception as exc:
                log.warning("Failed to sync metrics for %s/%s: %s", platform, post_id, exc)

    def was_product_posted(self, product_name: str, days_back: int = 30) -> bool:
        """Return True if this product was already used in the last N days."""
        cutoff = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
        with self._db() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM posts WHERE product_name=? AND posted_at > ?",
                (product_name, cutoff),
            ).fetchone()
        return (row[0] or 0) > 0

    def get_best_niches(self, top_n: int = 3) -> List[NichePerformance]:
        """Return the top-performing niches ranked by average views."""
        with self._db() as conn:
            rows = conn.execute(
                """
                SELECT niche,
                       COUNT(*) as total_posts,
                       AVG(views) as avg_views,
                       AVG(likes) as avg_likes,
                       AVG(shares) as avg_shares,
                       SUM(revenue_usd) as total_revenue
                FROM posts
                GROUP BY niche
                ORDER BY avg_views DESC
                LIMIT ?
                """,
                (top_n,),
            ).fetchall()

        results: List[NichePerformance] = []
        for row in rows:
            niche = row["niche"]
            top_hook = self._get_top_hook(niche)
            best_time = self._get_best_post_time(niche)
            results.append(NichePerformance(
                niche=niche,
                total_posts=row["total_posts"],
                avg_views=round(row["avg_views"] or 0, 0),
                avg_likes=round(row["avg_likes"] or 0, 0),
                avg_shares=round(row["avg_shares"] or 0, 0),
                top_hook=top_hook,
                best_post_time=best_time,
                total_revenue=round(row["total_revenue"] or 0, 2),
            ))
        return results

    def get_best_posting_times(self) -> Dict[str, List[str]]:
        """
        Return best posting times per platform based on historical engagement.
        Falls back to sensible defaults if no data yet.
        """
        defaults = {
            "tiktok": ["07:00", "12:00", "19:00"],
            "instagram": ["08:00", "13:00", "18:00"],
        }
        # With enough data we'd query the DB for highest-engagement hours.
        # For now return defaults (extend this after ~30 days of data).
        return defaults

    def weekly_report(self) -> str:
        """Generate a plain-text weekly performance summary."""
        cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()
        with self._db() as conn:
            summary = conn.execute(
                """
                SELECT platform,
                       COUNT(*) as posts,
                       SUM(views) as total_views,
                       SUM(likes) as total_likes,
                       SUM(shares) as total_shares,
                       SUM(revenue_usd) as revenue
                FROM posts
                WHERE posted_at > ?
                GROUP BY platform
                """,
                (cutoff,),
            ).fetchall()

        if not summary:
            return "No posts in the last 7 days yet."

        lines = ["=" * 50, "  WEEKLY PERFORMANCE REPORT", "=" * 50]
        total_rev = 0.0
        for row in summary:
            lines.append(f"\nPlatform : {row['platform'].upper()}")
            lines.append(f"Posts    : {row['posts']}")
            lines.append(f"Views    : {row['total_views']:,}")
            lines.append(f"Likes    : {row['total_likes']:,}")
            lines.append(f"Shares   : {row['total_shares']:,}")
            lines.append(f"Revenue  : ${row['revenue']:.2f}")
            total_rev += row["revenue"] or 0

        lines.append(f"\nTotal estimated revenue this week: ${total_rev:.2f}")
        lines.append("=" * 50)
        best = self.get_best_niches(3)
        if best:
            lines.append("\nTop niches:")
            for n in best:
                lines.append(f"  {n.niche}: avg {n.avg_views:,.0f} views, ${n.total_revenue:.2f} earned")
        return "\n".join(lines)

    # ── Platform metric fetchers ───────────────────────────────────────────

    def _fetch_tiktok_metrics(self, post_id: str) -> Optional[Dict]:
        if not config.TIKTOK_ACCESS_TOKEN:
            return None
        resp = requests.post(
            TIKTOK_VIDEO_QUERY_URL,
            headers={
                "Authorization": f"Bearer {config.TIKTOK_ACCESS_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "filters": {"video_ids": [post_id]},
                "fields": ["view_count", "like_count", "comment_count", "share_count"],
            },
            params={"fields": "view_count,like_count,comment_count,share_count"},
            timeout=15,
        )
        resp.raise_for_status()
        videos = resp.json().get("data", {}).get("videos", [])
        if not videos:
            return None
        v = videos[0]
        return {
            "views": v.get("view_count", 0),
            "likes": v.get("like_count", 0),
            "comments": v.get("comment_count", 0),
            "shares": v.get("share_count", 0),
            "saves": 0,
        }

    def _fetch_instagram_metrics(self, media_id: str) -> Optional[Dict]:
        if not config.INSTAGRAM_ACCESS_TOKEN:
            return None
        url = INSTAGRAM_MEDIA_URL.format(media_id=media_id)
        resp = requests.get(
            url,
            params={
                "fields": "like_count,comments_count,saved,reach,plays",
                "access_token": config.INSTAGRAM_ACCESS_TOKEN,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "views": data.get("plays", 0) or data.get("reach", 0),
            "likes": data.get("like_count", 0),
            "comments": data.get("comments_count", 0),
            "shares": 0,
            "saves": data.get("saved", 0),
        }

    # ── DB helpers ─────────────────────────────────────────────────────────

    @contextmanager
    def _db(self):
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS posts (
                    platform TEXT NOT NULL,
                    post_id TEXT NOT NULL,
                    product_name TEXT,
                    niche TEXT,
                    hook TEXT,
                    hashtags TEXT,
                    video_path TEXT,
                    posted_at TEXT,
                    views INTEGER DEFAULT 0,
                    likes INTEGER DEFAULT 0,
                    comments INTEGER DEFAULT 0,
                    shares INTEGER DEFAULT 0,
                    saves INTEGER DEFAULT 0,
                    affiliate_clicks INTEGER DEFAULT 0,
                    revenue_usd REAL DEFAULT 0.0,
                    last_synced_at TEXT,
                    PRIMARY KEY (platform, post_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_niche ON posts(niche)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_product ON posts(product_name)
            """)

    def _get_unsynced_posts(self) -> List[sqlite3.Row]:
        cutoff = (datetime.utcnow() - timedelta(hours=6)).isoformat()
        with self._db() as conn:
            return conn.execute(
                "SELECT platform, post_id FROM posts WHERE last_synced_at < ? OR last_synced_at IS NULL OR last_synced_at = ''",
                (cutoff,),
            ).fetchall()

    def _update_metrics(self, post_id: str, platform: str, metrics: Dict):
        with self._db() as conn:
            conn.execute(
                """
                UPDATE posts SET
                    views=?, likes=?, comments=?, shares=?, saves=?,
                    last_synced_at=?
                WHERE post_id=? AND platform=?
                """,
                (
                    metrics.get("views", 0), metrics.get("likes", 0),
                    metrics.get("comments", 0), metrics.get("shares", 0),
                    metrics.get("saves", 0),
                    datetime.utcnow().isoformat(),
                    post_id, platform,
                ),
            )

    def _get_top_hook(self, niche: str) -> str:
        with self._db() as conn:
            row = conn.execute(
                "SELECT hook FROM posts WHERE niche=? ORDER BY views DESC LIMIT 1",
                (niche,),
            ).fetchone()
        return row["hook"] if row else ""

    def _get_best_post_time(self, niche: str) -> str:
        """Return the hour of day that gets the most views for this niche."""
        with self._db() as conn:
            row = conn.execute(
                """
                SELECT strftime('%H:00', posted_at) as hour, AVG(views) as avg_views
                FROM posts WHERE niche=? AND views > 0
                GROUP BY hour ORDER BY avg_views DESC LIMIT 1
                """,
                (niche,),
            ).fetchone()
        return row["hour"] if row else "19:00"
