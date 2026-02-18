"""
Trend Analyzer
--------------
Identifies WHAT is trending RIGHT NOW so every video has the best
chance of going viral.

Sources used (all free / no paid API required):
  1. Google Trends (via pytrends) – rising search volume by keyword
  2. Amazon Movers & Shakers (web scrape) – best-selling products today
  3. Reddit trending posts (r/BuyItForLife, r/shutupandtakemymoney, etc.)
  4. Hardcoded viral hashtag pools per niche (always-fresh fallback)

Output: TrendReport with ranked products + trending hashtags for each niche.
"""

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Proven high-traffic hashtag pools per niche
# (updated quarterly – rotate for freshness)
NICHE_HASHTAGS: Dict[str, List[str]] = {
    "home organization": [
        "homeorganization", "organizationhacks", "cleantok",
        "tidyingtips", "homehacks", "cleaningmotivation",
        "organizationideas", "homesweethome", "declutter",
        "organizedliving", "tidyhome", "cleanwithme",
        "homeinspo", "minimalisthome", "storagehacks",
    ],
    "kitchen gadgets": [
        "kitchengadgets", "kitchenhacks", "cookingtiktok",
        "kitchentips", "foodtok", "kitchentools",
        "cookinghacks", "kitchenmusthaves", "foodiefinds",
        "kitchenfinds", "mealprep", "kitchenorganization",
        "cookingtips", "kitchenessentials", "gadgetsreview",
    ],
    "beauty tools": [
        "beautytok", "skincareroutine", "glowup",
        "beautytools", "skincarecheck", "beautyhacks",
        "makeuptips", "skincarefind", "beautymustaves",
        "skincaretiktok", "beautyreview", "glowuproutine",
        "antiaging", "facemassage", "beautyfinds",
    ],
    "fitness equipment": [
        "fitnesstok", "homeworkout", "gymtok",
        "workoutmotivation", "fitnesstips", "homegym",
        "workouthacks", "fitnessfinds", "gymequipment",
        "workouttips", "fitcheck", "fitnessmotivation",
        "exercisetips", "fitnesslife", "healthylifestyle",
    ],
    "tech accessories": [
        "techtok", "techfinds", "gadgets",
        "techreview", "techhacks", "techmusthaves",
        "gadgetreview", "techlife", "techcheck",
        "productreview", "amazonfinds", "tiktokmademebuyit",
        "techessentials", "gadgetlover", "mustHavetech",
    ],
}

# Universal high-traffic tags to always include
UNIVERSAL_HASHTAGS = [
    "tiktokmademebuyit", "amazonfinds", "productreview",
    "musthave", "ugc", "productfinds", "viralproduct",
]

# Amazon Movers & Shakers categories with URL paths
AMAZON_MOVERS_URLS = {
    "home organization": "https://www.amazon.com/gp/movers-and-shakers/home-garden/",
    "kitchen gadgets": "https://www.amazon.com/gp/movers-and-shakers/kitchen/",
    "beauty tools": "https://www.amazon.com/gp/movers-and-shakers/beauty/",
    "fitness equipment": "https://www.amazon.com/gp/movers-and-shakers/sporting-goods/",
    "tech accessories": "https://www.amazon.com/gp/movers-and-shakers/electronics/",
}

# Subreddits with viral product discoveries
REDDIT_SOURCES = [
    "shutupandtakemymoney",
    "BuyItForLife",
    "malegrooming",
    "femalefashionadvice",
    "homeimprovement",
]


@dataclass
class TrendingProduct:
    name: str
    rank: int                          # 1 = hottest
    niche: str
    source: str                        # amazon_movers | reddit | google_trends
    url: str = ""
    price: str = ""
    trend_score: float = 0.0           # 0.0–1.0 normalised


@dataclass
class TrendReport:
    niche: str
    trending_products: List[TrendingProduct] = field(default_factory=list)
    trending_hashtags: List[str] = field(default_factory=list)
    best_posting_times: List[str] = field(default_factory=lambda: ["07:00", "12:00", "19:00"])


class TrendAnalyzer:
    """Identifies trending products and hashtags for each configured niche."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    # ── Public API ─────────────────────────────────────────────────────────

    def analyze(self, niche: str) -> TrendReport:
        """Return a TrendReport for a single niche."""
        report = TrendReport(niche=niche)

        # Hashtags (always succeeds – no API needed)
        report.trending_hashtags = self._get_hashtags(niche)

        # Product trends
        products: List[TrendingProduct] = []

        # 1. Amazon Movers & Shakers
        try:
            products.extend(self._scrape_amazon_movers(niche))
        except Exception as exc:
            log.warning("Amazon Movers scrape failed for '%s': %s", niche, exc)

        # 2. Reddit trending
        try:
            products.extend(self._scrape_reddit(niche))
        except Exception as exc:
            log.warning("Reddit scrape failed for '%s': %s", niche, exc)

        # Deduplicate and rank by trend_score
        seen: set = set()
        unique: List[TrendingProduct] = []
        for p in sorted(products, key=lambda x: x.trend_score, reverse=True):
            key = p.name.lower()[:40]
            if key not in seen:
                seen.add(key)
                unique.append(p)

        # Re-rank
        for i, p in enumerate(unique):
            p.rank = i + 1

        report.trending_products = unique[:10]  # top 10
        log.info(
            "TrendReport for '%s': %d products, %d hashtags",
            niche, len(report.trending_products), len(report.trending_hashtags),
        )
        return report

    def analyze_all(self, niches: List[str]) -> Dict[str, TrendReport]:
        """Analyze all niches. Returns dict keyed by niche name."""
        reports: Dict[str, TrendReport] = {}
        for niche in niches:
            reports[niche] = self.analyze(niche)
            time.sleep(random.uniform(2.0, 4.0))  # polite delay
        return reports

    def get_top_hashtags(self, niche: str, count: int = 15) -> List[str]:
        """Quick access to top hashtags for a niche."""
        tags = self._get_hashtags(niche)
        return tags[:count]

    # ── Scrapers ───────────────────────────────────────────────────────────

    def _scrape_amazon_movers(self, niche: str) -> List[TrendingProduct]:
        url = AMAZON_MOVERS_URLS.get(niche)
        if not url:
            return []

        resp = self.session.get(url, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        products: List[TrendingProduct] = []
        items = soup.select(".zg-item-immersion, .p13n-sc-uncoverable-faceout")

        for i, item in enumerate(items[:20]):
            try:
                name_el = item.select_one(".p13n-sc-truncate-desktop-type2, .p13n-sc-truncated")
                price_el = item.select_one(".p13n-sc-price, .a-price .a-offscreen")
                link_el = item.select_one("a.a-link-normal")

                if not name_el:
                    continue

                name = name_el.get_text(strip=True)[:80]
                price = price_el.get_text(strip=True) if price_el else ""
                href = "https://www.amazon.com" + link_el["href"] if link_el else ""

                # Items at the top of Movers & Shakers get higher scores
                trend_score = max(0.1, 1.0 - (i * 0.05))

                products.append(TrendingProduct(
                    name=name,
                    rank=i + 1,
                    niche=niche,
                    source="amazon_movers",
                    url=href,
                    price=price,
                    trend_score=trend_score,
                ))
            except Exception:
                continue

        log.info("Amazon Movers: %d products for '%s'", len(products), niche)
        return products

    def _scrape_reddit(self, niche: str) -> List[TrendingProduct]:
        """Find top Reddit posts mentioning products in this niche."""
        products: List[TrendingProduct] = []
        keyword_map = {
            "home organization": ["organizer", "storage", "drawer"],
            "kitchen gadgets": ["kitchen gadget", "cooking tool", "kitchen hack"],
            "beauty tools": ["skincare", "beauty tool", "face massage"],
            "fitness equipment": ["home gym", "resistance band", "workout"],
            "tech accessories": ["desk setup", "cable management", "gadget"],
        }
        keywords = keyword_map.get(niche, [niche])
        subreddit = random.choice(REDDIT_SOURCES)

        url = f"https://www.reddit.com/r/{subreddit}/top.json?limit=10&t=week"
        headers = {**HEADERS, "Accept": "application/json"}

        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        posts = resp.json().get("data", {}).get("children", [])

        for i, post_data in enumerate(posts):
            post = post_data.get("data", {})
            title = post.get("title", "")
            score = post.get("score", 0)

            if not any(kw.lower() in title.lower() for kw in keywords):
                continue

            products.append(TrendingProduct(
                name=title[:80],
                rank=i + 1,
                niche=niche,
                source="reddit",
                url=f"https://reddit.com{post.get('permalink', '')}",
                trend_score=min(1.0, score / 5000),
            ))

        log.info("Reddit: %d relevant posts for '%s'", len(products), niche)
        return products

    # ── Hashtag builder ────────────────────────────────────────────────────

    def _get_hashtags(self, niche: str) -> List[str]:
        """Return a shuffled mix of niche-specific + universal hashtags."""
        niche_tags = NICHE_HASHTAGS.get(niche, [])
        combined = niche_tags + UNIVERSAL_HASHTAGS
        # Shuffle to get variety across posts but keep top niche tags first
        top = niche_tags[:5]  # always include top 5 niche tags
        rest = [t for t in combined if t not in top]
        random.shuffle(rest)
        return top + rest
