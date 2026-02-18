"""
UGC Automation – Main Orchestrator (v2)
----------------------------------------
Full pipeline: Trend Research → Product Research → AI Script
              → ElevenLabs Video → Cloud Upload → Post → Track Analytics

Run once:        python main.py --run-now
Run scheduler:   python main.py --schedule
Dry run:         python main.py --run-now --dry-run
Weekly report:   python main.py --report
Sync metrics:    python main.py --sync-metrics
Setup check:     python main.py --setup-check
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import schedule

from analytics_tracker import AnalyticsTracker, PostRecord
from affiliate_manager import AffiliateManager
from cloud_storage import CloudStorage
from config import config
from product_researcher import ProductResearcher
from script_generator import ScriptGenerator
from social_poster import SocialPoster
from trend_analyzer import TrendAnalyzer
from video_creator import VideoCreator

# ── Logging ────────────────────────────────────────────────────────────────────
os.makedirs(config.LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(config.LOGS_DIR, f"ugc_{datetime.now():%Y%m%d}.log")
        ),
    ],
)
log = logging.getLogger("ugc.main")


def run_pipeline(dry_run: bool = False):
    """
    Full pipeline:
      Trends → Research → Script → Video → Upload → Post → Record
    """
    log.info("=" * 60)
    log.info("UGC PIPELINE STARTED  (dry_run=%s)", dry_run)
    log.info("=" * 60)

    # ── Init modules ───────────────────────────────────────────────────────
    tracker = AnalyticsTracker()
    researcher = ProductResearcher()
    trend_analyzer = TrendAnalyzer()
    generator = ScriptGenerator(analytics_tracker=tracker)
    creator = VideoCreator()
    poster = SocialPoster()
    affiliate = AffiliateManager()
    storage = CloudStorage()

    # ── 1. Trend research ──────────────────────────────────────────────────
    log.info("Step 1/5 – Analysing trends…")
    trend_reports = trend_analyzer.analyze_all(config.NICHES)

    # Pull trending hashtags for all niches (merged pool)
    all_trending_tags: list = []
    for report in trend_reports.values():
        all_trending_tags.extend(report.trending_hashtags[:5])
    all_trending_tags = list(dict.fromkeys(all_trending_tags))[:30]

    # ── 2. Product research (filter already-posted products) ───────────────
    log.info("Step 2/5 – Researching products…")
    raw_products = researcher.get_products_for_all_niches()

    # Skip products featured in the last 30 days
    products = [p for p in raw_products if not tracker.was_product_posted(p.name)]
    if not products:
        log.warning("All researched products were already featured recently. Using unfiltered list.")
        products = raw_products

    log.info("Using %d product(s)", len(products))

    # ── 3. Generate scripts ────────────────────────────────────────────────
    log.info("Step 3/5 – Generating scripts…")
    scripts = generator.generate_batch(products, trending_hashtags=all_trending_tags)
    log.info("Generated %d script(s)", len(scripts))

    # Inject affiliate links
    for script in scripts:
        if script.product.amazon_url:
            link = affiliate.build_link(script.product.amazon_url, script.product.niche)
            script.affiliate_url = link.affiliate_url
            # Append affiliate link to caption
            if link.affiliate_url and link.affiliate_url != script.product.amazon_url:
                script.caption += f"\n\nShop here: {link.affiliate_url}"

    # Save scripts to disk
    for script in scripts:
        _save_script(script)

    # ── 4. Create videos ───────────────────────────────────────────────────
    log.info("Step 4/5 – Creating videos…")
    video_pairs = []
    for script in scripts:
        try:
            path = creator.create(script)
            video_pairs.append((path, script))
            log.info("Video ready: %s", path)
        except Exception as exc:
            log.error("Video creation failed for '%s': %s", script.product.name, exc)

    # ── 5. Upload and post ─────────────────────────────────────────────────
    if dry_run:
        log.info("Step 5/5 – DRY RUN: skipping upload. Videos in %s", config.VIDEOS_DIR)
        _print_scripts_preview(scripts)
        return

    log.info("Step 5/5 – Uploading and posting…")
    for video_path, script in video_pairs:
        # Upload to cloud (needed for Instagram)
        try:
            public_url = storage.upload(video_path)
        except Exception as exc:
            log.error("Cloud upload failed: %s", exc)
            public_url = ""

        # Post to each platform
        results = poster.post(video_path, script, public_url=public_url)

        for r in results:
            if r.success:
                log.info("[%s] Posted – %s", r.platform, r.post_url)
                # Record in analytics
                tracker.record_post(PostRecord(
                    platform=r.platform,
                    post_id=r.post_id,
                    product_name=script.product.name,
                    niche=script.product.niche,
                    hook=script.hook,
                    hashtags=script.hashtags,
                    video_path=video_path,
                    posted_at=datetime.utcnow().isoformat(),
                ))
            else:
                log.error("[%s] Post failed – %s", r.platform, r.error)

    log.info("Pipeline complete.")


# ── Scheduler ──────────────────────────────────────────────────────────────────

def run_scheduler(dry_run: bool = False):
    """Schedule the pipeline to run at configured times every day."""
    # Use analytics to get best posting times
    tracker = AnalyticsTracker()
    best_times = tracker.get_best_posting_times()

    # Collect the union of best times across platforms, then fall back to config
    all_times = set()
    for platform_times in best_times.values():
        all_times.update(platform_times)
    post_times = sorted(all_times) if all_times else config.POST_TIMES

    log.info("Scheduler started. Posting at: %s", post_times)
    for t in post_times:
        schedule.every().day.at(t).do(run_pipeline, dry_run=dry_run)

    # Sync metrics every 6 hours
    schedule.every(6).hours.do(_sync_metrics)

    # Weekly report every Sunday at 9 AM
    schedule.every().sunday.at("09:00").do(_print_weekly_report)

    log.info("Scheduler running. Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(30)


def _sync_metrics():
    try:
        tracker = AnalyticsTracker()
        tracker.sync_metrics()
    except Exception as exc:
        log.error("Metric sync failed: %s", exc)


def _print_weekly_report():
    try:
        tracker = AnalyticsTracker()
        print(tracker.weekly_report())
    except Exception as exc:
        log.error("Report generation failed: %s", exc)


# ── CLI helpers ────────────────────────────────────────────────────────────────

def setup_check():
    """Print a checklist of what's configured and what's missing."""
    print("\n" + "="*60)
    print("  UGC AUTOMATION – SETUP CHECKLIST")
    print("="*60)

    checks = [
        ("ANTHROPIC_API_KEY", "Claude AI script generation (REQUIRED)"),
        ("TIKTOK_ACCESS_TOKEN", "Post to TikTok"),
        ("INSTAGRAM_ACCESS_TOKEN", "Post to Instagram"),
        ("INSTAGRAM_ACCOUNT_ID", "Post to Instagram"),
        ("ELEVENLABS_API_KEY", "Premium voice quality (optional but recommended)"),
        ("PEXELS_API_KEY", "Free stock video backgrounds (optional)"),
        ("AMAZON_AFFILIATE_TAG", "Amazon affiliate commissions"),
        ("TIKTOK_SHOP_AFFILIATE_ID", "TikTok Shop commissions"),
        ("AWS_ACCESS_KEY_ID", "S3/R2 cloud storage for Instagram"),
        ("B2_KEY_ID", "Backblaze B2 cloud storage (alternative to S3)"),
    ]

    all_good = True
    for env_key, label in checks:
        val = os.getenv(env_key, "")
        status = "✓" if val else "✗ MISSING"
        if not val:
            all_good = False
        print(f"  {status:<12} {label}")
        print(f"             env: {env_key}")

    print("="*60)
    if all_good:
        print("  All keys configured. Ready to run!")
    else:
        print("  Set missing keys in your .env file, then run again.")
    print()

    # Show affiliate programs setup guide
    affiliate = AffiliateManager()
    affiliate.print_setup_guide()


def _save_script(script):
    os.makedirs(config.SCRIPTS_DIR, exist_ok=True)
    safe_name = "".join(c if c.isalnum() else "_" for c in script.product.name)[:40]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(config.SCRIPTS_DIR, f"{safe_name}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(script.to_dict(), f, indent=2, ensure_ascii=False)
    log.info("Script saved: %s", path)


def _print_scripts_preview(scripts):
    print("\n" + "="*60)
    print("  SCRIPTS PREVIEW (dry run)")
    print("="*60)
    for s in scripts:
        print(f"\nProduct : {s.product.name}")
        print(f"Hook    : {s.hook}")
        print(f"CTA     : {s.cta}")
        print(f"Tags    : #{' #'.join(s.hashtags[:5])}")
    print("="*60 + "\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="UGC Automation – TikTok & Instagram auto-posting pipeline"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-now", action="store_true",
                       help="Run the full pipeline once immediately")
    group.add_argument("--schedule", action="store_true",
                       help="Start the daily scheduler")
    group.add_argument("--report", action="store_true",
                       help="Print this week's performance report")
    group.add_argument("--sync-metrics", action="store_true",
                       help="Fetch latest metrics from TikTok/Instagram APIs")
    group.add_argument("--setup-check", action="store_true",
                       help="Check which API keys are configured")

    parser.add_argument("--dry-run", action="store_true",
                        help="Skip upload/posting – only research + script + video")
    args = parser.parse_args()

    if args.run_now:
        run_pipeline(dry_run=args.dry_run)
    elif args.schedule:
        run_scheduler(dry_run=args.dry_run)
    elif args.report:
        _print_weekly_report()
    elif args.sync_metrics:
        _sync_metrics()
    elif args.setup_check:
        setup_check()


if __name__ == "__main__":
    main()
