import time
import anthropic
from utils.config import load_config, get_env
from utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a world-class AI art director with deep expertise in creating prompts
that produce fine art prints suitable for premium Etsy listings.

Your prompts must:
1. Produce images that look like genuine hand-painted artworks, not AI-generated photos
2. Specify painting medium and technique (oil on canvas, watercolour on paper, etc.)
3. Include lighting quality (soft diffused light, dramatic side lighting, golden hour glow)
4. Specify surface texture that reads in the final image (visible canvas texture,
   brushstroke direction, impasto thickness, paper grain)
5. Include compositional guidance (rule of thirds, centred focal point, dynamic diagonal)
6. Reference a specific real artistic tradition or named movement without copying a
   specific artist's style (Impressionist tradition, Dutch Golden Age still life
   tradition, Contemporary abstract expressionism)
7. End every prompt with technical quality markers:
   "photorealistic painting detail, ultra high resolution, 300 DPI print quality,
   museum quality fine art, Etsy wall art bestseller"
8. Never include: text, watermarks, signatures, frames, people's recognisable faces
   (unless portrait style), brand logos, or explicit content
9. Be written as a single flowing paragraph of 90 to 130 words
10. Be formatted for a 2:3 portrait aspect ratio (standard print proportions)

Return ONLY the prompt text. No commentary, preamble, or explanation."""

FORBIDDEN_WORDS = {"caption", "subtitle", "watermark", "signature", "label",
                   "title card", "lettering"}
# "text" alone is forbidden but "texture/textured/texturing" are fine — checked as whole word
_TEXT_STANDALONE = True
QUALITY_MARKERS = {"300 dpi", "ultra high resolution", "photorealistic painting detail",
                   "museum quality", "fine art", "print quality"}
MEDIUMS = {"oil", "watercolour", "watercolor", "acrylic", "gouache", "pastel"}
TEXTURE_WORDS = {"canvas texture", "brushstroke", "impasto", "paper grain", "palette knife"}
LIGHTING_WORDS = {"golden hour", "soft diffused", "dramatic side", "natural window", "studio softbox",
                  "atmospheric light", "warm light", "diffused light", "side lighting"}
COMPOSITION_WORDS = {"rule of thirds", "centred", "centered", "diagonal", "foreground", "focal point"}


def validate_prompt(prompt: str) -> tuple[bool, list[str]]:
    errors = []
    lower = prompt.lower()
    words = lower.split()

    word_count = len(words)
    if word_count < 80:
        errors.append(f"Prompt too short: {word_count} words (min 80)")

    import re
    for word in FORBIDDEN_WORDS:
        if re.search(r'\b' + re.escape(word) + r'\b', lower):
            errors.append(f"Forbidden word found: '{word}'")

    # "text" as a standalone word (not inside texture/textured/etc.)
    if re.search(r'\btext\b', lower):
        errors.append("Forbidden word found: 'text'")

    quality_found = sum(1 for m in QUALITY_MARKERS if m in lower)
    if quality_found < 2:
        errors.append(f"Not enough quality markers: found {quality_found}, need 2")

    if not any(m in lower for m in MEDIUMS):
        errors.append("No painting medium specified")

    if not any(t in lower for t in TEXTURE_WORDS):
        errors.append("No surface texture descriptor found")

    if not any(l in lower for l in LIGHTING_WORDS):
        errors.append("No lighting descriptor found")

    if not any(c in lower for c in COMPOSITION_WORDS):
        errors.append("No compositional descriptor found")

    return len(errors) == 0, errors


def enhance_prompt(
    style_name: str,
    style_base_description: str,
    colour_variation: str,
    technique_hint: str = "",
    dry_run: bool = False,
) -> str:
    if dry_run:
        logger.info("[DRY RUN] Returning mock enhanced prompt")
        return (
            f"A stunning {style_name.lower()} painting featuring {colour_variation.lower()}, "
            f"executed in oil on canvas with visible impasto brushstrokes and thick palette knife marks. "
            f"Soft diffused natural light illuminates the composition, following the rule of thirds "
            f"with a strong centred focal point. The canvas texture is prominently visible beneath "
            f"layers of richly textured paint in the Contemporary abstract expressionism tradition. "
            f"Portrait 2:3 format composition with dynamic diagonal energy. "
            f"Photorealistic painting detail, ultra high resolution, 300 DPI print quality, "
            f"museum quality fine art, Etsy wall art bestseller."
        )

    config = load_config()
    api_key = get_env("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    user_message = (
        f"Style: {style_name}\n"
        f"Base description: {style_base_description}\n"
        f"Colour variation: {colour_variation}\n"
        f"Technique emphasis: {technique_hint}\n\n"
        f"Generate an optimised image generation prompt for this painting."
    )

    max_retries = config.get("max_retries_per_stage", 2)
    backoff = config.get("retry_backoff_seconds", 15)

    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Stage 1: Enhancing prompt (attempt {attempt + 1})")
            start = time.time()

            response = client.messages.create(
                model=config.get("claude_model", "claude-sonnet-4-6"),
                max_tokens=config.get("prompt_max_tokens", 400),
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            elapsed = time.time() - start
            prompt = response.content[0].text.strip()
            logger.info(f"Stage 1: Claude responded in {elapsed:.1f}s, {len(prompt.split())} words")

            valid, errors = validate_prompt(prompt)
            if valid:
                logger.info("Stage 1: Prompt validation passed")
                return prompt

            logger.warning(f"Stage 1: Prompt validation failed: {errors}")

            if attempt < max_retries:
                stricter_note = (
                    f"\n\nYour previous prompt had these issues: {'; '.join(errors)}. "
                    f"Please fix them and try again."
                )
                user_message_retry = user_message + stricter_note
                time.sleep(backoff)
                user_message = user_message_retry
            else:
                logger.error("Stage 1: Max retries reached, using last prompt despite validation errors")
                return prompt

        except anthropic.RateLimitError:
            logger.warning("Stage 1: Rate limit hit — waiting 60s")
            time.sleep(60)
        except Exception as e:
            logger.error(f"Stage 1: Unexpected error: {e}")
            if attempt < max_retries:
                time.sleep(backoff)
            else:
                raise

    raise RuntimeError("Stage 1: Failed to generate a valid prompt after all retries")
