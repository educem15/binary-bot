# UGC Automation – Faceless TikTok & Instagram Channel

Fully automated pipeline that researches trending products, writes UGC scripts,
creates short-form vertical videos, and posts them to TikTok and Instagram Reels
on a schedule — **no face required, runs unattended**.

```
Product Research → AI Script → Video (TTS + stock footage) → Auto-post
```

---

## How It Works

| Step | Module | What it does |
|------|--------|-------------|
| 1 | `product_researcher.py` | Scrapes trending products from Amazon / seed list |
| 2 | `script_generator.py` | Claude AI writes hook + body + CTA + caption + hashtags |
| 3 | `video_creator.py` | TTS voiceover + Pexels stock video + burned captions → MP4 |
| 4 | `social_poster.py` | Uploads to TikTok Content API + Instagram Graph API |
| – | `main.py` | Orchestrates everything; supports `--run-now` and `--schedule` |

---

## Quick Start

### 1. Install dependencies

```bash
cd ugc_automation
pip install -r requirements.txt

# Also install ImageMagick (required by MoviePy for text overlays)
# Ubuntu/Debian:
sudo apt-get install imagemagick
# macOS:
brew install imagemagick
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in your keys (see sections below)
source .env
```

### 3. Test with a dry run (no upload)

```bash
python main.py --run-now --dry-run
```

Videos and scripts are saved to `output/` for review.

### 4. Run for real

```bash
python main.py --run-now
```

### 5. Start the daily scheduler

```bash
# Posts at 8 AM, 12 PM, 6 PM by default (configurable in config.py)
python main.py --schedule
```

Run it in the background with `nohup` or set it up as a system service:

```bash
nohup python main.py --schedule > output/logs/scheduler.log 2>&1 &
```

---

## API Keys You Need

### Required
- **Anthropic API key** – https://console.anthropic.com/
  Powers script generation. ~$0.01–$0.05 per script.

### For posting (at least one)
- **TikTok Content Posting API**
  1. Create app at https://developers.tiktok.com/
  2. Enable the "Content Posting API" product
  3. Complete OAuth2 to get `TIKTOK_ACCESS_TOKEN`

- **Instagram Graph API**
  1. Create app at https://developers.facebook.com/
  2. Connect Instagram Professional (Creator or Business) account
  3. Generate a long-lived Page Access Token
  4. Get your `INSTAGRAM_ACCOUNT_ID` from the API

### Optional (improves quality)
- **ElevenLabs** (https://elevenlabs.io) – much more natural voices. Free tier available.
- **Pexels API** (https://www.pexels.com/api/) – free stock video. Without it, a plain background is used.

---

## Customise Your Niche

Edit `config.py`:

```python
NICHES: List[str] = [
    "home organization",
    "kitchen gadgets",
    "beauty tools",
    "fitness equipment",
    "tech accessories",
]
```

Change or add any niches relevant to your channel.

---

## Output Files

```
output/
├── scripts/    # JSON files with every generated script (for review/audit)
├── videos/     # Finished MP4 files
└── logs/       # Daily log files
```

---

## Monetisation Tips

1. **Affiliate links** – Add your Amazon Associate tag in `.env` → links auto-appended to descriptions
2. **TikTok Shop** – Apply for TikTok Shop Creator and add products directly
3. **Brand deals** – Once you have 1k+ followers, brands will pay $50–$500 per post
4. **Consistency** – The scheduler posts 3× per day; aim for 90 days straight

---

## Architecture Overview

```
main.py
  │
  ├─► ProductResearcher   (scrapes Amazon / uses seed list)
  │       │
  │       └─► ProductInfo  (name, description, benefits, price)
  │
  ├─► ScriptGenerator    (calls Claude API)
  │       │
  │       └─► VideoScript  (hook, body, CTA, caption, hashtags, keywords)
  │
  ├─► VideoCreator       (TTS + Pexels + MoviePy)
  │       │
  │       └─► MP4 file   (1080×1920, ≤45 sec)
  │
  └─► SocialPoster       (TikTok Content API + Instagram Graph API)
          │
          └─► PostResult  (success/failure, post URL)
```
