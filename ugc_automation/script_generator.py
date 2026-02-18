"""
Script Generator (v2 – Sales Psychology Edition)
-------------------------------------------------
Uses the Anthropic Claude API to write short, viral UGC video scripts.

v2 improvements:
  - Sales psychology triggers: FOMO, social proof, loss aversion, curiosity gap
  - 3 hook variants generated per product (A/B/C test)
  - Top-performer hook selected based on analytics data
  - ElevenLabs UGC voice hints embedded in voiceover text (pacing cues)
  - Trend-aware hashtags injected from TrendAnalyzer
  - Affiliate link woven naturally into CTA
"""

import json
import logging
import re
import random
from dataclasses import dataclass, field
from typing import List, Optional

import anthropic

from config import config
from product_researcher import ProductInfo

log = logging.getLogger(__name__)


@dataclass
class VideoScript:
    product: ProductInfo
    hook: str = ""
    hook_variants: List[str] = field(default_factory=list)  # A/B/C alternatives
    body: str = ""
    cta: str = ""
    caption: str = ""
    hashtags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    full_voiceover: str = ""
    affiliate_url: str = ""   # injected by AffiliateManager before posting
    selected_hook_index: int = 0  # tracked for A/B analytics

    def to_dict(self) -> dict:
        return {
            "product_name": self.product.name,
            "niche": self.product.niche,
            "hook": self.hook,
            "hook_variants": self.hook_variants,
            "body": self.body,
            "cta": self.cta,
            "caption": self.caption,
            "hashtags": self.hashtags,
            "keywords": self.keywords,
            "full_voiceover": self.full_voiceover,
            "affiliate_url": self.affiliate_url,
            "selected_hook_index": self.selected_hook_index,
        }


# ── Sales psychology system prompt ────────────────────────────────────────────
SYSTEM_PROMPT = """You are a world-class UGC (User Generated Content) scriptwriter for TikTok and Instagram Reels.
Your scripts generate real sales. You use proven psychological triggers naturally and conversationally:

TRIGGERS YOU USE:
  - Curiosity gap: "I can't believe I waited so long to find this…"
  - Social proof: "This has 50,000 five-star reviews for a reason…"
  - Loss aversion / FOMO: "I almost missed this deal…" / "It keeps selling out…"
  - Specificity: Exact numbers feel real. "$12" beats "cheap". "3 seconds" beats "fast".
  - Relatability: Start with a problem the viewer has. Make them say "that's me!"
  - Contrast: Before vs after. Life without this vs with it.
  - Scarcity (honest): "The price goes up on Friday" / "Almost out of stock"

RULES:
  - Faceless channel: voiceover only, no face, no personal references to appearance
  - Never sound like a commercial. Sound like a friend telling another friend
  - Hook must make the viewer STOP scrolling in the first 2 seconds
  - Under 45 seconds total when spoken at 130 words/minute
  - CTA always includes where to buy (link in bio / TikTok Shop)
  - Body covers exactly 2 benefits with specifics, not vague adjectives
"""

SCRIPT_PROMPT_TEMPLATE = """Write a high-converting UGC video script for this product.

Product: {name}
Niche: {niche}
Description: {description}
Key benefits: {benefits}
Price range: {price_range}
Top-performing hook style for this niche: {best_hook_style}

Return ONLY a valid JSON object with these exact keys (no extra text):
{{
  "hook_variants": [
    "Hook A – curiosity gap style (e.g. 'I can't believe I lived without this for 30 years…')",
    "Hook B – social proof style (e.g. 'This thing has 80,000 five-star reviews and now I know why…')",
    "Hook C – problem/relatable style (e.g. 'If your [relevant problem], you NEED to see this…')"
  ],
  "body": "Main script: problem → benefit 1 with specific detail → benefit 2 with specific detail. Conversational. 55–75 words.",
  "cta": "Call to action – tell them exactly where to buy and create urgency (1 sentence, max 20 words)",
  "caption": "Full ready-to-paste caption: opening statement + 2–3 benefit bullets + emoji + soft CTA. 60–100 words.",
  "hashtags": ["10", "to", "15", "relevant", "hashtags", "no", "hash", "symbol"],
  "keywords": ["3 to 5 keywords for stock footage search, specific scenes like 'clean kitchen counter' or 'woman workout home'"]
}}
"""

# Niche-specific best hook styles (informed by platform data)
BEST_HOOK_STYLES = {
    "home organization": "problem-relatable ('if your drawers look like mine used to…')",
    "kitchen gadgets": "curiosity gap ('I've been wasting 20 minutes doing this the wrong way…')",
    "beauty tools": "social proof ('dermatologists have been recommending this for years…')",
    "fitness equipment": "before/after ('I lost 10 lbs in 30 days just by adding this to my routine…')",
    "tech accessories": "specificity ('this $18 thing completely changed my WFH setup…')",
}


class ScriptGenerator:
    """Generates high-converting UGC video scripts using Claude."""

    def __init__(self, analytics_tracker=None):
        if not config.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file or environment variables."
            )
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.analytics = analytics_tracker  # optional – used to select best hook

    def generate(
        self,
        product: ProductInfo,
        trending_hashtags: Optional[List[str]] = None,
    ) -> VideoScript:
        """Generate a complete script with 3 hook variants for A/B testing."""
        best_hook_style = BEST_HOOK_STYLES.get(product.niche, "curiosity gap")

        prompt = SCRIPT_PROMPT_TEMPLATE.format(
            name=product.name,
            niche=product.niche,
            description=product.description,
            benefits=", ".join(product.key_benefits) or "highly rated, affordable",
            price_range=product.price_range,
            best_hook_style=best_hook_style,
        )

        log.info("Generating script for: %s", product.name)

        message = self.client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = message.content[0].text.strip()
        data = self._parse_json(raw)

        script = VideoScript(product=product)
        script.hook_variants = data.get("hook_variants", [])
        script.hook = self._select_best_hook(script.hook_variants, product.niche)
        script.body = data.get("body", "")
        script.cta = data.get("cta", "")
        script.caption = data.get("caption", "")
        script.keywords = data.get("keywords", [product.niche])

        # Merge generated hashtags with trending hashtags from TrendAnalyzer
        generated_tags = data.get("hashtags", [])
        extra_tags = trending_hashtags or []
        combined = list(dict.fromkeys(generated_tags + extra_tags))  # dedupe, preserve order
        script.hashtags = combined[:20]

        script.full_voiceover = f"{script.hook} {script.body} {script.cta}".strip()

        log.info(
            "Script generated – hook: '%s…' (%d words total)",
            script.hook[:40],
            len(script.full_voiceover.split()),
        )
        return script

    def generate_batch(
        self,
        products: List[ProductInfo],
        trending_hashtags: Optional[List[str]] = None,
    ) -> List[VideoScript]:
        scripts = []
        for product in products:
            try:
                scripts.append(self.generate(product, trending_hashtags))
            except Exception as exc:
                log.error("Failed to generate script for '%s': %s", product.name, exc)
        return scripts

    # ── Hook selection ─────────────────────────────────────────────────────

    def _select_best_hook(self, variants: List[str], niche: str) -> str:
        """
        Pick the hook variant to use.
        - If analytics data is available, pick the style that historically performs best.
        - Otherwise rotate through variants (A → B → C → A…) for natural A/B testing.
        """
        if not variants:
            return ""

        if self.analytics:
            top_hook = self.analytics._get_top_hook(niche)
            if top_hook:
                # Find which variant style matches the top-performing hook
                for i, v in enumerate(variants):
                    if any(word in v.lower() for word in top_hook.lower().split()[:3]):
                        log.info("Analytics: selected hook variant %d for niche '%s'", i, niche)
                        return v

        # Fallback: random selection (distributes A/B/C evenly over time)
        chosen = random.choice(variants)
        log.info("Selected hook variant randomly")
        return chosen

    # ── JSON parser ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> dict:
        cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"Could not parse JSON from Claude response:\n{text}")
