"""
UGC Automation – Main Orchestrator
------------------------------------
Run once:      python main.py --run-now
Run scheduler: python main.py --schedule
Dry run:       python main.py --run-now --dry-run   (no upload, saves video locally)
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import schedule

from config import config
from product_researcher import ProductResearcher
from script_generator import ScriptGenerator
from social_poster import SocialPoster
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
      Research → Script → Video → Post
    """
    log.info("=" * 60)
    log.info("UGC pipeline started  (dry_run=%s)", dry_run)
    log.info("=" * 60)

    researcher = ProductResearcher()
    generator = ScriptGenerator()
    creator = VideoCreator()
    poster = SocialPoster()

    # ── 1. Research products ───────────────────────────────────────────────
    log.info("Step 1/4 – Researching products…")
    products = researcher.get_products_for_all_niches()
    log.info("Found %d product(s)", len(products))

    if not products:
        log.error("No products found – aborting.")
        return

    # ── 2. Generate scripts ────────────────────────────────────────────────
    log.info("Step 2/4 – Generating scripts…")
    scripts = generator.generate_batch(products)
    log.info("Generated %d script(s)", len(scripts))

    # Save scripts to disk for reference
    for script in scripts:
        _save_script(script)

    # ── 3. Create videos ───────────────────────────────────────────────────
    log.info("Step 3/4 – Creating videos…")
    video_paths = []
    for script in scripts:
        try:
            path = creator.create(script)
            video_paths.append((path, script))
            log.info("Video ready: %s", path)
        except Exception as exc:
            log.error("Video creation failed for '%s': %s", script.product.name, exc)

    # ── 4. Post to social media ────────────────────────────────────────────
    if dry_run:
        log.info("Step 4/4 – DRY RUN: skipping upload. Videos saved to %s", config.VIDEOS_DIR)
        return

    log.info("Step 4/4 – Posting to social media…")
    for video_path, script in video_paths:
        results = poster.post(video_path, script)
        for r in results:
            if r.success:
                log.info("[%s] Posted successfully – %s", r.platform, r.post_url)
            else:
                log.error("[%s] Post failed – %s", r.platform, r.error)

    log.info("Pipeline complete.")


def _save_script(script):
    """Persist script JSON for review / audit trail."""
    os.makedirs(config.SCRIPTS_DIR, exist_ok=True)
    safe_name = "".join(c if c.isalnum() else "_" for c in script.product.name)[:40]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(config.SCRIPTS_DIR, f"{safe_name}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(script.to_dict(), f, indent=2, ensure_ascii=False)
    log.info("Script saved: %s", path)


def run_scheduler(dry_run: bool = False):
    """Schedule the pipeline to run at configured times every day."""
    log.info("Scheduler started. Posting times: %s", config.POST_TIMES)

    for t in config.POST_TIMES:
        schedule.every().day.at(t).do(run_pipeline, dry_run=dry_run)
        log.info("Scheduled daily run at %s", t)

    log.info("Waiting for next scheduled run… (Ctrl+C to stop)")
    while True:
        schedule.run_pending()
        time.sleep(30)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="UGC Automation – TikTok & Instagram content pipeline"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run-now", action="store_true",
        help="Run the full pipeline once immediately"
    )
    group.add_argument(
        "--schedule", action="store_true",
        help="Start the scheduler (runs at POST_TIMES configured in config.py)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Research + script + video, but do NOT upload to any platform"
    )
    args = parser.parse_args()

    if args.run_now:
        run_pipeline(dry_run=args.dry_run)
    elif args.schedule:
        run_scheduler(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
