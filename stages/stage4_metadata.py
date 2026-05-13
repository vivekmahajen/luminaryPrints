import json
import time
import anthropic
from utils.config import load_config, get_env
from utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a senior Etsy SEO specialist with proven experience growing digital art shops
to 5-figure monthly revenue. You write listings that rank in Etsy search, convert
browsers into buyers, and generate 5-star reviews.

You will receive details about a digitally created art print and must write a complete
Etsy listing in JSON format.

Etsy SEO rules you must follow:
1. Title: exactly 120-140 characters including spaces, starts with the most searchable
   keyword, uses "|" as a separator between keyword clusters, no ALL CAPS
2. Description: opens with a hook sentence (benefit to the buyer), includes all
   practical details (sizes, file format, DPI, what's included), ends with a
   call to action, uses short paragraphs and bullet points for scannability
3. Tags: exactly 13 tags, each under 20 characters, mix of broad category tags
   (printable-wall-art, digital-download) and specific style tags, no tag should
   repeat a word already in the title
4. Price: recommend $9.99 for digital download single listing, $14.99 for
   "complete bundle" framing
5. Never make claims that cannot be verified (e.g. "award-winning", "best on Etsy")
6. Never mention AI generation in the listing — describe it as "digitally created art"

Return ONLY valid JSON. No markdown fences, no commentary."""

DESCRIPTION_TEMPLATE_HINT = """Structure the description as follows:
[Hook — one sentence benefit to the buyer]

[What this print looks like — 2 sentences of evocative description]

✨ WHAT YOU GET:
• [Number] high-resolution JPEG files
• Sizes included: [list all sizes]
• Resolution: 300 DPI — sharp at any size
• Format: JPEG, ready to send to any print shop

🖨️ HOW TO PRINT:
• Home printer — select "fit to page" or "actual size"
• Local print shop — Staples, FedEx Office, Walgreens Photo
• Online — Shutterfly, Snapfish, Canvaspop

🎁 PERFECT FOR:
• [Room types — living room, bedroom, etc.]
• [Occasion — housewarming, birthday, etc.]
• [Style — minimalist, boho, modern, etc.]

⚡ INSTANT DOWNLOAD — your files are available immediately after payment.
No physical item is shipped. Frame not included.

Questions? Message me and I'll reply within 24 hours."""


def _validate_metadata(data: dict) -> tuple[bool, list[str]]:
    errors = []

    title = data.get("title", "")
    if not (100 <= len(title) <= 140):
        errors.append(f"Title length {len(title)} not in 100-140 range")
    if title and title[:2].isupper() and title[1] == title[1].upper():
        pass  # Single letter OK, check for ALL CAPS words
    for word in title.split():
        if len(word) > 3 and word.isupper():
            errors.append(f"ALL CAPS word in title: '{word}'")
    if title and title[0].lower() in ("a ", "th") and len(title) > 4 and title[:4].lower() == "the ":
        errors.append("Title starts with 'The' — wastes SEO space")

    etsy_kws = {"printable wall art", "digital download", "instant download",
                "wall art print", "digital print"}
    title_lower = title.lower()
    if not any(kw in title_lower for kw in etsy_kws):
        errors.append("Title missing Etsy keyword (printable wall art, digital download, etc.)")

    tags = data.get("tags", [])
    if len(tags) != 13:
        errors.append(f"Tags count: {len(tags)} (must be exactly 13)")
    for tag in tags:
        if len(tag) > 20:
            errors.append(f"Tag too long (>{len(tag)} chars): '{tag}'")

    desc = data.get("description", "")
    if not (400 <= len(desc) <= 2000):
        errors.append(f"Description length {len(desc)} not in 400-2000 range")
    desc_lower = desc.lower()
    if "300 dpi" not in desc_lower:
        errors.append("Description missing '300 DPI'")
    if "instant download" not in desc_lower and "digital download" not in desc_lower:
        errors.append("Description missing download mention")
    if "no physical item" not in desc_lower and "no physical" not in desc_lower:
        errors.append("Description missing 'no physical item is shipped' disclaimer")

    return len(errors) == 0, errors


def generate_metadata(
    style_name: str,
    colour_variation: str,
    enhanced_prompt: str,
    sizes_generated: list[str],
    dry_run: bool = False,
) -> dict:
    if dry_run:
        logger.info("[DRY RUN] Returning mock Etsy metadata")
        return _mock_metadata(style_name, colour_variation, sizes_generated)

    config = load_config()
    api_key = get_env("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    sizes_str = ", ".join(sizes_generated)
    user_message = (
        f"Style: {style_name}\n"
        f"Colour variation: {colour_variation}\n"
        f"Image description: {enhanced_prompt}\n"
        f"Print sizes included: {sizes_str}\n\n"
        f"Description structure to follow:\n{DESCRIPTION_TEMPLATE_HINT}\n\n"
        f"Return the full Etsy listing as valid JSON matching this schema:\n"
        + json.dumps(_schema_example(), indent=2)
    )

    max_retries = config.get("max_retries_per_stage", 2)
    backoff = config.get("retry_backoff_seconds", 15)

    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Stage 4: Generating Etsy metadata (attempt {attempt + 1})")
            start = time.time()

            response = client.messages.create(
                model=config.get("claude_model", "claude-sonnet-4-6"),
                max_tokens=config.get("metadata_max_tokens", 800),
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            elapsed = time.time() - start
            raw = response.content[0].text.strip()
            logger.info(f"Stage 4: Claude responded in {elapsed:.1f}s")

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(f"Stage 4: JSON parse failed, raw response: {raw[:200]}")
                if attempt < max_retries:
                    user_message = (
                        "Your previous response could not be parsed as JSON. "
                        "Return ONLY raw JSON, no markdown, no explanation.\n\n"
                        + user_message
                    )
                    time.sleep(backoff)
                    continue
                raise ValueError("Stage 4: Could not parse Claude response as JSON")

            valid, errors = _validate_metadata(data)
            if valid:
                logger.info("Stage 4: Metadata validation passed")
                return data

            logger.warning(f"Stage 4: Metadata validation errors: {errors}")
            if attempt < max_retries:
                user_message = (
                    f"Your previous JSON had these issues: {'; '.join(errors)}. "
                    f"Fix them and return the corrected JSON only.\n\n"
                    + user_message
                )
                time.sleep(backoff)
            else:
                logger.error("Stage 4: Using last metadata despite validation errors")
                return data

        except anthropic.RateLimitError:
            logger.warning("Stage 4: Rate limit — waiting 60s")
            time.sleep(60)
        except Exception as e:
            logger.error(f"Stage 4: Unexpected error: {e}")
            if attempt < max_retries:
                time.sleep(backoff)
            else:
                raise

    raise RuntimeError("Stage 4: Failed to generate valid metadata after all retries")


def _schema_example() -> dict:
    return {
        "title": "string — 120-140 chars",
        "description": "string — full multi-paragraph listing body",
        "tags": ["array", "of", "exactly", "13", "strings"],
        "materials": ["Digital file", "Instant download", "JPEG format"],
        "suggested_price_digital": 9.99,
        "suggested_price_bundle_3": 19.99,
        "suggested_price_physical_8x10": 24.99,
        "suggested_price_physical_16x20": 44.99,
        "suggested_price_physical_24x36": 64.99,
        "category_path": "Art & Collectibles > Prints > Digital Prints",
        "section_name": "string",
        "occasion_tags": [],
        "style_tags": [],
        "room_tags": [],
        "colour_tags": [],
        "thumbnail_advice": "one sentence",
        "pinterest_caption": "one sentence",
        "instagram_caption": "2 sentences + 5 hashtags",
        "listing_photos_needed": [],
    }


def _build_mock_title(style_name: str, colour_variation: str) -> str:
    variation_short = colour_variation[:25].strip()
    base = f"Printable Wall Art {style_name} Print | Digital Download | {variation_short} Wall Decor"
    if len(base) < 100:
        base = f"Printable Wall Art {style_name} | Instant Digital Download | {variation_short} Home Wall Decor"
    if len(base) > 140:
        base = base[:140]
    return base


def _mock_metadata(style_name: str, colour_variation: str, sizes: list[str]) -> dict:
    sizes_str = ", ".join(sizes)
    return {
        "title": _build_mock_title(style_name, colour_variation),
        "description": (
            f"Transform your walls with this stunning {style_name.lower()} print.\n\n"
            f"Rich tones of {colour_variation.lower()} create a sophisticated atmosphere "
            f"that elevates any room.\n\n"
            f"✨ WHAT YOU GET:\n• {len(sizes)} high-resolution JPEG files\n"
            f"• Sizes included: {sizes_str}\n"
            f"• Resolution: 300 DPI — sharp at any size\n"
            f"• Format: JPEG, ready to send to any print shop\n\n"
            f"🖨️ HOW TO PRINT:\n• Home printer — select \"fit to page\"\n"
            f"• Local print shop — Staples, FedEx Office, Walgreens Photo\n"
            f"• Online — Shutterfly, Snapfish, Canvaspop\n\n"
            f"🎁 PERFECT FOR:\n• Living room, bedroom, home office\n"
            f"• Housewarming gift, birthday present\n"
            f"• Minimalist, boho, modern interiors\n\n"
            f"⚡ INSTANT DOWNLOAD — your files are available immediately after payment.\n"
            f"No physical item is shipped. Frame not included.\n\n"
            f"Questions? Message me and I'll reply within 24 hours."
        ),
        "tags": [
            "printable-wall-art",
            "digital-download",
            "instant-download",
            "wall-art-print",
            "digital-print",
            "abstract-art-print",
            "living-room-art",
            "bedroom-decor",
            "boho-wall-art",
            "modern-home-decor",
            "large-wall-art",
            "minimalist-print",
            "art-gift",
        ],
        "materials": ["Digital file", "Instant download", "JPEG format"],
        "suggested_price_digital": 9.99,
        "suggested_price_bundle_3": 19.99,
        "suggested_price_physical_8x10": 24.99,
        "suggested_price_physical_16x20": 44.99,
        "suggested_price_physical_24x36": 64.99,
        "category_path": "Art & Collectibles > Prints > Digital Prints",
        "section_name": style_name,
        "occasion_tags": ["housewarming", "birthday"],
        "style_tags": ["modern", "minimalist", "boho"],
        "room_tags": ["living-room", "bedroom", "home-office"],
        "colour_tags": [colour_variation.split(",")[0].strip().lower().replace(" ", "-")],
        "thumbnail_advice": "Show the print in a styled room mockup with a natural wood frame.",
        "pinterest_caption": f"Elevate your home with this {style_name.lower()} wall art — instant digital download.",
        "instagram_caption": (
            f"New drop: {style_name} art print in {colour_variation.lower()}. "
            f"Download and print instantly — link in bio. "
            f"#wallart #printableart #homedecor #digitaldownload #artprint"
        ),
        "listing_photos_needed": [
            "Styled room mockup — living room with natural frame",
            "Close-up texture detail",
            "Size comparison chart",
            "Multiple frames mockup",
        ],
    }
