"""
Script Generator
----------------
Uses the Anthropic Claude API to write short, viral UGC video scripts.

Each script has:
  - Hook      (first 1–3 sec – grabs attention)
  - Body      (product benefits, 10–30 sec)
  - CTA       (call-to-action, 3–5 sec)
  - Caption   (ready-to-paste social media caption with hashtags)
  - Keywords  (for stock video search)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List

import anthropic

from config import config
from product_researcher import ProductInfo

log = logging.getLogger(__name__)


@dataclass
class VideoScript:
    product: ProductInfo
    hook: str = ""
    body: str = ""
    cta: str = ""
    caption: str = ""
    hashtags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)  # for stock video search
    full_voiceover: str = ""                            # hook + body + cta combined

    def to_dict(self) -> dict:
        return {
            "product_name": self.product.name,
            "niche": self.product.niche,
            "hook": self.hook,
            "body": self.body,
            "cta": self.cta,
            "caption": self.caption,
            "hashtags": self.hashtags,
            "keywords": self.keywords,
            "full_voiceover": self.full_voiceover,
        }


SYSTEM_PROMPT = """You are an expert UGC (User Generated Content) scriptwriter for TikTok and Instagram Reels.
Your scripts are conversational, authentic, and designed for a faceless channel (no face needed – just voiceover + product footage).
You write in an engaging, relatable tone. You never sound like an ad.
You always follow the Hook → Problem/Benefit → CTA structure.
You keep the total voiceover under 45 seconds when spoken at a natural pace (~130 words/min).
"""

SCRIPT_PROMPT_TEMPLATE = """Write a viral UGC video script for the following product.

Product name: {name}
Niche: {niche}
Description: {description}
Key benefits: {benefits}
Price range: {price_range}

Return ONLY a valid JSON object with these exact keys:
{{
  "hook": "The opening line (1–2 sentences, max 15 words, must create curiosity or shock)",
  "body": "The main script covering 2–3 key benefits (conversational, 60–80 words)",
  "cta": "Call to action (1 sentence, e.g. 'Link in bio – grab it before it sells out')",
  "caption": "Full social media caption (3–5 sentences + emoji, ready to paste)",
  "hashtags": ["list", "of", "10", "relevant", "hashtags", "without", "the", "#", "symbol"],
  "keywords": ["3 to 5 keywords to search for stock video footage, e.g. 'kitchen', 'cooking', 'clean home'"]
}}
"""


class ScriptGenerator:
    """Generates UGC video scripts using Claude."""

    def __init__(self):
        if not config.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file or environment variables."
            )
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def generate(self, product: ProductInfo) -> VideoScript:
        """Generate a complete script for the given product."""
        prompt = SCRIPT_PROMPT_TEMPLATE.format(
            name=product.name,
            niche=product.niche,
            description=product.description,
            benefits=", ".join(product.key_benefits) or "highly rated, affordable",
            price_range=product.price_range,
        )

        log.info("Generating script for: %s", product.name)

        message = self.client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = message.content[0].text.strip()
        data = self._parse_json(raw)

        script = VideoScript(product=product)
        script.hook = data.get("hook", "")
        script.body = data.get("body", "")
        script.cta = data.get("cta", "")
        script.caption = data.get("caption", "")
        script.hashtags = data.get("hashtags", [])
        script.keywords = data.get("keywords", [product.niche])
        script.full_voiceover = f"{script.hook} {script.body} {script.cta}".strip()

        log.info("Script generated (%d words)", len(script.full_voiceover.split()))
        return script

    def generate_batch(self, products: List[ProductInfo]) -> List[VideoScript]:
        scripts = []
        for product in products:
            try:
                scripts.append(self.generate(product))
            except Exception as exc:
                log.error("Failed to generate script for '%s': %s", product.name, exc)
        return scripts

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract JSON from Claude's response (handles markdown fences)."""
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract just the JSON object
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"Could not parse JSON from Claude response:\n{text}")
