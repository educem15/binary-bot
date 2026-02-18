"""
Affiliate Link Manager
-----------------------
Manages affiliate links across multiple programs so every video
drives trackable, commission-earning traffic.

Programs supported:
  1. Amazon Associates      – up to 10% commission, millions of products
  2. TikTok Shop Affiliate  – 5–20% commission, in-app checkout
  3. ShareASale             – thousands of brands
  4. Impact.com             – premium brands (Airbnb, Canva, etc.)
  5. ClickBank              – digital products (high commissions, 30–75%)
  6. LTK (LikeToKnowIt)    – lifestyle / fashion creators

How to join each program:
  • Amazon Associates:  https://affiliate-program.amazon.com/
  • TikTok Shop:        https://shop.tiktok.com/business/en/seller-university (Creator Centre → Affiliate)
  • ShareASale:         https://www.shareasale.com/signup/
  • Impact:             https://impact.com/
  • ClickBank:          https://www.clickbank.com/
  • LTK:               https://www.shopltk.com/

Payment methods:
  • Amazon – direct bank transfer / gift card, monthly (60-day delay)
  • TikTok Shop – PayPal / bank, bi-weekly
  • ShareASale – check / direct deposit, monthly ($50 threshold)
  • Impact – PayPal / bank, monthly
  • ClickBank – check / wire, bi-weekly
  • LTK – PayPal, monthly
"""

import logging
import os
import urllib.parse
from dataclasses import dataclass, field
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class AffiliateProgram:
    name: str
    commission_range: str           # e.g. "5–10%"
    cookie_duration: str            # e.g. "24 hours"
    payment_threshold: str          # e.g. "$10"
    payment_frequency: str          # e.g. "monthly"
    sign_up_url: str
    env_key: str                    # environment variable holding the tag/ID
    tag: str = ""                   # populated at runtime from env


# Catalogue of programs with realistic commission data
PROGRAMS: Dict[str, AffiliateProgram] = {
    "amazon": AffiliateProgram(
        name="Amazon Associates",
        commission_range="1–10% (varies by category)",
        cookie_duration="24 hours",
        payment_threshold="$10",
        payment_frequency="Monthly (60-day delay)",
        sign_up_url="https://affiliate-program.amazon.com/",
        env_key="AMAZON_AFFILIATE_TAG",
    ),
    "tiktok_shop": AffiliateProgram(
        name="TikTok Shop Affiliate",
        commission_range="5–20%",
        cookie_duration="In-app session",
        payment_threshold="$20",
        payment_frequency="Bi-weekly",
        sign_up_url="https://shop.tiktok.com/business/en",
        env_key="TIKTOK_SHOP_AFFILIATE_ID",
    ),
    "shareasale": AffiliateProgram(
        name="ShareASale",
        commission_range="5–30% (brand-specific)",
        cookie_duration="30–90 days",
        payment_threshold="$50",
        payment_frequency="Monthly (20th)",
        sign_up_url="https://www.shareasale.com/signup/",
        env_key="SHAREASALE_AFFILIATE_ID",
    ),
    "impact": AffiliateProgram(
        name="Impact.com",
        commission_range="5–30% (brand-specific)",
        cookie_duration="30 days",
        payment_threshold="$10",
        payment_frequency="Monthly",
        sign_up_url="https://impact.com/",
        env_key="IMPACT_AFFILIATE_ID",
    ),
    "clickbank": AffiliateProgram(
        name="ClickBank",
        commission_range="30–75% (digital products)",
        cookie_duration="60 days",
        payment_threshold="$10",
        payment_frequency="Bi-weekly",
        sign_up_url="https://www.clickbank.com/",
        env_key="CLICKBANK_AFFILIATE_ID",
    ),
    "ltk": AffiliateProgram(
        name="LTK (LikeToKnowIt)",
        commission_range="3–20%",
        cookie_duration="30 days",
        payment_threshold="$0",
        payment_frequency="Monthly",
        sign_up_url="https://www.shopltk.com/",
        env_key="LTK_AFFILIATE_ID",
    ),
}

# Maps product niches to the best affiliate programs for that niche
NICHE_BEST_PROGRAMS = {
    "home organization": ["amazon", "shareasale"],
    "kitchen gadgets": ["amazon", "shareasale"],
    "beauty tools": ["amazon", "ltk", "shareasale"],
    "fitness equipment": ["amazon", "impact", "shareasale"],
    "tech accessories": ["amazon", "impact"],
}


@dataclass
class AffiliateLink:
    original_url: str
    affiliate_url: str
    program: str
    short_url: str = ""          # optional – set after URL shortening
    commission_range: str = ""


class AffiliateManager:
    """Builds affiliate-tagged links for products across multiple programs."""

    def __init__(self):
        # Load env tags into program objects
        for prog in PROGRAMS.values():
            prog.tag = os.getenv(prog.env_key, "")

    # ── Public API ─────────────────────────────────────────────────────────

    def build_link(self, product_url: str, niche: str) -> AffiliateLink:
        """
        Build the best affiliate link for a product URL + niche combination.
        Falls back gracefully if no programs are configured.
        """
        # Determine which programs are active (have a tag set) for this niche
        preferred = NICHE_BEST_PROGRAMS.get(niche, ["amazon"])
        for prog_key in preferred:
            prog = PROGRAMS.get(prog_key)
            if prog and prog.tag:
                aff_url = self._build_url(product_url, prog_key, prog.tag)
                if aff_url:
                    return AffiliateLink(
                        original_url=product_url,
                        affiliate_url=aff_url,
                        program=prog.name,
                        commission_range=prog.commission_range,
                    )

        # No affiliate program configured – return original URL unchanged
        log.warning(
            "No affiliate program configured for niche '%s' – returning original URL", niche
        )
        return AffiliateLink(
            original_url=product_url,
            affiliate_url=product_url,
            program="none",
        )

    def build_all_links(self, product_url: str, niche: str) -> List[AffiliateLink]:
        """Build links for ALL active programs (useful for bio link aggregators)."""
        links: List[AffiliateLink] = []
        for prog_key, prog in PROGRAMS.items():
            if not prog.tag:
                continue
            aff_url = self._build_url(product_url, prog_key, prog.tag)
            if aff_url:
                links.append(AffiliateLink(
                    original_url=product_url,
                    affiliate_url=aff_url,
                    program=prog.name,
                    commission_range=prog.commission_range,
                ))
        return links

    def list_unconfigured(self) -> List[AffiliateProgram]:
        """Returns programs that haven't been configured yet (missing env var)."""
        return [p for p in PROGRAMS.values() if not p.tag]

    def print_setup_guide(self):
        """Print a one-time setup guide to terminal."""
        unconfigured = self.list_unconfigured()
        if not unconfigured:
            print("[AffiliateManager] All programs are configured.")
            return
        print("\n" + "="*60)
        print("  AFFILIATE PROGRAMS – Action Required")
        print("="*60)
        for prog in unconfigured:
            print(f"\n  {prog.name}")
            print(f"  Commission: {prog.commission_range}")
            print(f"  Cookie:     {prog.cookie_duration}")
            print(f"  Pay:        {prog.payment_frequency} (min {prog.payment_threshold})")
            print(f"  Sign up:    {prog.sign_up_url}")
            print(f"  Then set:   export {prog.env_key}=<your_tag>")
        print("\n" + "="*60 + "\n")

    # ── Link builders per platform ─────────────────────────────────────────

    def _build_url(self, base_url: str, program: str, tag: str) -> Optional[str]:
        try:
            parsed = urllib.parse.urlparse(base_url)
            qs = urllib.parse.parse_qs(parsed.query)

            if program == "amazon":
                qs["tag"] = [tag]
                new_qs = urllib.parse.urlencode(qs, doseq=True)
                return urllib.parse.urlunparse(parsed._replace(query=new_qs))

            elif program == "tiktok_shop":
                # TikTok shop links use a referral parameter
                qs["referrer"] = [tag]
                new_qs = urllib.parse.urlencode(qs, doseq=True)
                return urllib.parse.urlunparse(parsed._replace(query=new_qs))

            elif program in ("shareasale", "impact", "clickbank", "ltk"):
                # These use redirect through their own tracking links
                # The user will supply the pre-built tracking URL from their dashboard
                # We append their ID as a sub-ID for analytics
                qs["sub_id"] = [tag]
                new_qs = urllib.parse.urlencode(qs, doseq=True)
                return urllib.parse.urlunparse(parsed._replace(query=new_qs))

        except Exception as exc:
            log.warning("Failed to build %s affiliate URL: %s", program, exc)

        return None
