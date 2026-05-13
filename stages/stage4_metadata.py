import json
import time
import anthropic
from utils.config import load_config, get_env
from utils.logger import get_logger

logger = get_logger(__name__)

# Ask Claude only for the creative fields — keeps JSON small and reliable.
# Python assembles the full listing from these parts.
SYSTEM_PROMPT = """You are a senior Etsy SEO specialist. You write listings that rank in Etsy
search and convert browsers into buyers.

Rules:
1. title: 120-140 characters, starts with the most searchable keyword, uses | as separator,
   no ALL CAPS words, must contain "printable wall art" OR "digital download" OR "wall art print"
2. hook: one compelling sentence that opens the description (benefit to the buyer, no "This is a...")
3. visual_description: 2 sentences describing what the print looks like evocatively
4. room_uses: 3 bullet points listing ideal rooms and occasions (plain text, no emoji)
5. tags: exactly 13 tags, each tag 20 characters or fewer, mix of broad and specific,
   include "printable-wall-art", "digital-download", "instant-download"

Return ONLY a JSON object with these five keys. No markdown, no explanation."""


def _validate_core(data: dict) -> tuple[bool, list[str]]:
    errors = []

    title = data.get("title", "")
    if not (100 <= len(title) <= 140):
        errors.append(f"Title length {len(title)} — must be 100-140 chars")
    for word in title.split():
        if len(word) > 3 and word.isupper():
            errors.append(f"ALL CAPS word in title: '{word}'")
    etsy_kws = {"printable wall art", "digital download", "instant download",
                "wall art print", "digital print"}
    if not any(kw in title.lower() for kw in etsy_kws):
        errors.append("Title missing Etsy keyword")

    tags = data.get("tags", [])
    if len(tags) != 13:
        errors.append(f"Need exactly 13 tags, got {len(tags)}")
    for tag in tags:
        if len(tag) > 20:
            errors.append(f"Tag too long ({len(tag)} chars): '{tag}'")

    if not data.get("hook"):
        errors.append("Missing hook sentence")

    return len(errors) == 0, errors


def _build_description(data: dict, sizes: list[str], style_name: str) -> str:
    sizes_str = ", ".join(sizes)
    hook = data.get("hook", "")
    visual = data.get("visual_description", "")
    room_uses = data.get("room_uses", "Living room, bedroom, home office.")

    return (
        f"{hook}\n\n"
        f"{visual}\n\n"
        f"✨ WHAT YOU GET:\n"
        f"• {len(sizes)} high-resolution JPEG files\n"
        f"• Sizes included: {sizes_str}\n"
        f"• Resolution: 300 DPI — sharp at any size\n"
        f"• Format: JPEG, ready to send to any print shop\n\n"
        f"\U0001f5a8️ HOW TO PRINT:\n"
        f"• Home printer — select fit to page or actual size\n"
        f"• Local print shop — Staples, FedEx Office, Walgreens Photo\n"
        f"• Online — Shutterfly, Snapfish, Canvaspop\n\n"
        f"\U0001f381 PERFECT FOR:\n"
        f"{room_uses}\n\n"
        f"⚡ INSTANT DOWNLOAD — your files are available immediately after payment.\n"
        f"No physical item is shipped. Frame not included.\n\n"
        f"Questions? Message me and I'll reply within 24 hours."
    )


def _assemble_full_metadata(core: dict, sizes: list[str], style_name: str, colour_variation: str) -> dict:
    description = _build_description(core, sizes, style_name)
    colour_tag = colour_variation.split(",")[0].strip().lower().replace(" ", "-")[:20]
    return {
        "title": core["title"],
        "description": description,
        "tags": core["tags"],
        "materials": ["Digital file", "Instant download", "JPEG format"],
        "suggested_price_digital": 9.99,
        "suggested_price_bundle_3": 19.99,
        "suggested_price_physical_8x10": 24.99,
        "suggested_price_physical_16x20": 44.99,
        "suggested_price_physical_24x36": 64.99,
        "category_path": "Art & Collectibles > Prints > Digital Prints",
        "section_name": style_name,
        "occasion_tags": ["housewarming", "birthday", "new-home"],
        "style_tags": ["modern", "minimalist", "contemporary"],
        "room_tags": ["living-room", "bedroom", "home-office"],
        "colour_tags": [colour_tag],
        "thumbnail_advice": "Show the print in a styled room mockup with a natural wood frame.",
        "pinterest_caption": (
            f"Elevate your home with this {style_name.lower()} wall art "
            f"— instant digital download."
        ),
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


def _validate_metadata(data: dict) -> tuple[bool, list[str]]:
    """Validate the fully assembled metadata dict."""
    errors = []

    title = data.get("title", "")
    if not (100 <= len(title) <= 140):
        errors.append(f"Title length {len(title)} not in 100-140 range")
    etsy_kws = {"printable wall art", "digital download", "instant download",
                "wall art print", "digital print"}
    if not any(kw in title.lower() for kw in etsy_kws):
        errors.append("Title missing Etsy keyword")

    tags = data.get("tags", [])
    if len(tags) != 13:
        errors.append(f"Tags count: {len(tags)} (must be exactly 13)")
    for tag in tags:
        if len(tag) > 20:
            errors.append(f"Tag too long ({len(tag)} chars): '{tag}'")

    desc = data.get("description", "")
    if not (400 <= len(desc) <= 2000):
        errors.append(f"Description length {len(desc)} not in 400-2000 range")
    desc_lower = desc.lower()
    if "300 dpi" not in desc_lower:
        errors.append("Description missing '300 DPI'")
    if "instant download" not in desc_lower and "digital download" not in desc_lower:
        errors.append("Description missing download mention")
    if "no physical item" not in desc_lower and "no physical" not in desc_lower:
        errors.append("Description missing 'no physical item' disclaimer")

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

    user_message = (
        f"Style: {style_name}\n"
        f"Colour variation: {colour_variation}\n"
        f"Image description: {enhanced_prompt}\n"
        f"Print sizes included: {', '.join(sizes_generated)}\n\n"
        f"Return JSON with exactly these keys: title, hook, visual_description, room_uses, tags"
    )

    max_retries = config.get("max_retries_per_stage", 2)
    backoff = config.get("retry_backoff_seconds", 15)

    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Stage 4: Generating Etsy metadata (attempt {attempt + 1})")
            start = time.time()

            response = client.messages.create(
                model=config.get("claude_model", "claude-sonnet-4-6"),
                max_tokens=config.get("metadata_max_tokens", 2048),
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            elapsed = time.time() - start
            raw = response.content[0].text.strip()
            # Strip markdown fences if Claude adds them despite instructions
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            logger.info(f"Stage 4: Claude responded in {elapsed:.1f}s")

            try:
                core = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.warning(f"Stage 4: JSON parse failed ({e}) — raw: {raw[:300]}")
                if attempt < max_retries:
                    user_message = (
                        "Your previous response could not be parsed as JSON. "
                        "Return ONLY a raw JSON object with keys: "
                        "title, hook, visual_description, room_uses, tags. No markdown.\n\n"
                        + user_message
                    )
                    time.sleep(backoff)
                    continue
                raise ValueError("Stage 4: Could not parse Claude response as JSON")

            valid, errors = _validate_core(core)
            if not valid:
                logger.warning(f"Stage 4: Core validation errors: {errors}")
                if attempt < max_retries:
                    user_message = (
                        f"Issues with your previous response: {'; '.join(errors)}. "
                        f"Fix and return corrected JSON only.\n\n" + user_message
                    )
                    time.sleep(backoff)
                    continue
                logger.error("Stage 4: Using last response despite validation errors")

            metadata = _assemble_full_metadata(core, sizes_generated, style_name, colour_variation)
            logger.info("Stage 4: Metadata assembled successfully")
            return metadata

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


def _build_mock_title(style_name: str, colour_variation: str) -> str:
    variation_short = colour_variation[:30].strip()
    candidates = [
        f"Printable Wall Art {style_name} Print | Digital Download | {variation_short} Wall Decor",
        f"Printable Wall Art {style_name} | Instant Digital Download | {variation_short} Home Decor",
        f"Digital Download Wall Art Print | {style_name} | {variation_short} | Instant Printable Decor",
    ]
    for title in candidates:
        if 100 <= len(title) <= 140:
            return title
    # Pad or truncate to fit
    title = candidates[1]
    if len(title) < 100:
        title = title + " | Large Wall Art"
    return title[:140]


def _mock_metadata(style_name: str, colour_variation: str, sizes: list[str]) -> dict:
    core = {
        "title": _build_mock_title(style_name, colour_variation),
        "hook": f"Transform your walls with this stunning {style_name.lower()} art print.",
        "visual_description": (
            f"Rich tones of {colour_variation.lower()} create a sophisticated atmosphere. "
            f"Thick painterly marks and vibrant colour give this piece gallery-worthy presence."
        ),
        "room_uses": (
            "• Living room, bedroom, home office\n"
            "• Housewarming gift, birthday present\n"
            "• Minimalist, boho, modern interiors"
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
    }
    return _assemble_full_metadata(core, sizes, style_name, colour_variation)
