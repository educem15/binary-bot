"""
Product Researcher
------------------
Finds trending products to feature in UGC videos.

Sources:
  1. TikTok trending hashtags (web scrape / unofficial)
  2. Amazon trending / best sellers (via BeautifulSoup)
  3. Manual seed list fallback

Returns a list of ProductInfo dicts ready for the script generator.
"""

import logging
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

from config import config

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class ProductInfo:
    name: str
    description: str
    niche: str
    price_range: str = "under $30"
    key_benefits: List[str] = field(default_factory=list)
    amazon_url: str = ""
    image_url: str = ""
    source: str = "manual"


# ── Fallback seed products (used when scraping fails) ─────────────────────────
SEED_PRODUCTS = {
    "home organization": [
        ProductInfo(
            name="Stackable Drawer Organizer Set",
            description="Modular clear plastic organizers that stack to save space in any drawer.",
            niche="home organization",
            price_range="$15–$25",
            key_benefits=["saves drawer space", "see-through design", "dishwasher safe"],
            source="seed",
        ),
        ProductInfo(
            name="Magnetic Spice Rack",
            description="Wall-mounted magnetic jars that turn any fridge or backsplash into a spice station.",
            niche="home organization",
            price_range="$20–$35",
            key_benefits=["frees counter space", "airtight jars", "strong magnets"],
            source="seed",
        ),
    ],
    "kitchen gadgets": [
        ProductInfo(
            name="Avocado Slicer 3-in-1",
            description="Splits, pits, and slices avocados in seconds without a knife.",
            niche="kitchen gadgets",
            price_range="$8–$15",
            key_benefits=["safer than a knife", "BPA-free", "dishwasher safe"],
            source="seed",
        ),
        ProductInfo(
            name="Clip-On Pot Strainer",
            description="Silicone strainer that clips to any pot – no colander needed.",
            niche="kitchen gadgets",
            price_range="$10–$18",
            key_benefits=["saves washing a colander", "heat-resistant", "fits any pot"],
            source="seed",
        ),
    ],
    "beauty tools": [
        ProductInfo(
            name="Gua Sha Facial Tool",
            description="Rose quartz stone that reduces puffiness and sculpts the face with daily massage.",
            niche="beauty tools",
            price_range="$12–$25",
            key_benefits=["reduces puffiness", "improves circulation", "natural stone"],
            source="seed",
        ),
        ProductInfo(
            name="LED Face Mask",
            description="7-colour light therapy mask for acne, anti-aging, and skin brightening at home.",
            niche="beauty tools",
            price_range="$30–$60",
            key_benefits=["dermatologist approved", "no downtime", "multiple modes"],
            source="seed",
        ),
    ],
    "fitness equipment": [
        ProductInfo(
            name="Resistance Band Set",
            description="5 colour-coded fabric bands for full-body home workouts.",
            niche="fitness equipment",
            price_range="$15–$28",
            key_benefits=["portable", "no gym needed", "multiple resistance levels"],
            source="seed",
        ),
        ProductInfo(
            name="Jump Rope with Counter",
            description="Weighted speed rope with digital rep counter built into the handles.",
            niche="fitness equipment",
            price_range="$20–$35",
            key_benefits=["burns 10 cal/min", "adjustable length", "tracks reps automatically"],
            source="seed",
        ),
    ],
    "tech accessories": [
        ProductInfo(
            name="MagSafe Wallet Stand",
            description="Slim card holder that attaches magnetically and props up your phone hands-free.",
            niche="tech accessories",
            price_range="$15–$30",
            key_benefits=["holds 3 cards", "doubles as a stand", "strong magnet"],
            source="seed",
        ),
        ProductInfo(
            name="Cable Management Box",
            description="Wooden box that hides power strips and cable clutter on any desk.",
            niche="tech accessories",
            price_range="$25–$40",
            key_benefits=["hides ugly cables", "bamboo design", "fits most power strips"],
            source="seed",
        ),
    ],
}


class ProductResearcher:
    """Finds and returns trending products for a given niche."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    # ── Public API ─────────────────────────────────────────────────────────

    def get_products(self, niche: str, count: int = 2) -> List[ProductInfo]:
        """Return `count` products for the given niche."""
        products: List[ProductInfo] = []

        # Try Amazon trending first
        try:
            products = self._scrape_amazon_movers(niche, count)
            if products:
                log.info("Amazon scrape returned %d products for '%s'", len(products), niche)
        except Exception as exc:
            log.warning("Amazon scrape failed (%s) – using seed products", exc)

        # Fill any gap with seed data
        if len(products) < count:
            seed = self._get_seed_products(niche)
            products.extend(seed[: count - len(products)])

        return products[:count]

    def get_products_for_all_niches(self) -> List[ProductInfo]:
        """Cycle through all configured niches and collect products."""
        all_products: List[ProductInfo] = []
        for niche in config.NICHES:
            all_products.extend(self.get_products(niche, count=1))
            time.sleep(random.uniform(1.5, 3.0))  # polite delay
        return all_products[: config.PRODUCTS_PER_RUN]

    # ── Scrapers ───────────────────────────────────────────────────────────

    def _scrape_amazon_movers(self, niche: str, count: int) -> List[ProductInfo]:
        """Scrape Amazon's 'Movers & Shakers' for a niche keyword."""
        search_term = niche.replace(" ", "+")
        url = f"https://www.amazon.com/s?k={search_term}&s=review-rank"

        resp = self.session.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        products: List[ProductInfo] = []
        cards = soup.select('[data-component-type="s-search-result"]')

        for card in cards[:count * 2]:  # grab extras in case some fail
            try:
                name_el = card.select_one("h2 .a-text-normal")
                price_el = card.select_one(".a-price .a-offscreen")
                img_el = card.select_one(".s-image")
                link_el = card.select_one("h2 a")

                if not name_el:
                    continue

                name = name_el.get_text(strip=True)[:80]
                price = price_el.get_text(strip=True) if price_el else "See on Amazon"
                img = img_el["src"] if img_el else ""
                href = "https://www.amazon.com" + link_el["href"] if link_el else ""

                if config.AMAZON_AFFILIATE_TAG and href:
                    href += f"?tag={config.AMAZON_AFFILIATE_TAG}"

                products.append(ProductInfo(
                    name=name,
                    description=f"Trending product in {niche}: {name}",
                    niche=niche,
                    price_range=price,
                    key_benefits=[f"highly rated in {niche}", "trending on Amazon"],
                    amazon_url=href,
                    image_url=img,
                    source="amazon",
                ))

                if len(products) >= count:
                    break

            except Exception:
                continue

        return products

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_seed_products(self, niche: str) -> List[ProductInfo]:
        pool = SEED_PRODUCTS.get(niche, [])
        if not pool:
            # Generic fallback
            pool = [
                ProductInfo(
                    name=f"Trending {niche.title()} Product",
                    description=f"A viral product from the {niche} niche.",
                    niche=niche,
                    price_range="under $30",
                    key_benefits=["affordable", "trending", "highly rated"],
                    source="generic_seed",
                )
            ]
        return random.sample(pool, k=min(len(pool), 2))
