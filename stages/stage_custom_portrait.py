import time
import anthropic
from pathlib import Path
from utils.config import load_config, get_env
from utils.logger import get_logger

logger = get_logger(__name__)

PORTRAIT_TYPES = {
    "pet": {
        "label": "Custom Pet Portrait",
        "style_id": "custom_pet_portrait",
        "system_prompt": """You are an expert AI art director specialising in pet portrait paintings.
Generate a prompt that transforms a pet photo into a premium fine art print.

Rules:
1. Describe the subject as "the pet in the reference photo" — never assume breed
2. Specify an expressive painterly style: oil on canvas, impasto, palette knife, or watercolour
3. Include atmospheric background that complements the pet: soft bokeh, abstract colour wash, impressionist garden
4. Specify dramatic or soft lighting that flatters the subject
5. Use a centred composition with the pet as the clear focal point
6. End with: "photorealistic painting detail, ultra high resolution, 300 DPI print quality, museum quality fine art"
7. Never mention AI, photography, or digital — describe as a hand-painted portrait
8. 80-120 words, single paragraph, portrait 2:3 format

Return ONLY the prompt text.""",
    },
    "faceless": {
        "label": "Custom Faceless Portrait",
        "style_id": "custom_faceless_portrait",
        "system_prompt": """You are an expert AI art director specialising in contemporary faceless portrait paintings.
Generate a prompt that transforms a person photo into an abstract faceless fine art print.

Rules:
1. Describe the subject as "a figure" or "a silhouette" — no facial features rendered
2. Emphasise clothing, posture, hair, and setting rather than facial details
3. Specify an expressive painterly style: contemporary oil, gestural acrylic, or mixed media
4. Use a striking background: colour wash, abstract shapes, or impressionist environment
5. Include dramatic lighting: golden hour backlight, soft studio diffusion, or bold contrast
6. Centred or rule-of-thirds composition with figure as focal point
7. End with: "photorealistic painting detail, ultra high resolution, 300 DPI print quality, museum quality fine art"
8. 80-120 words, single paragraph, portrait 2:3 format

Return ONLY the prompt text.""",
    },
}

STYLE_VARIATIONS = {
    "pet": [
        "Impressionist oil painting, soft warm background, golden hour light",
        "Bold impasto style, vibrant colour palette, textured canvas",
        "Watercolour wash, delicate botanical background, soft diffused light",
        "Contemporary acrylic, abstract colour field background, dramatic contrast",
        "Loose painterly brushwork, moody dark background, spotlight lighting",
        "Pastel tones, dreamy soft-focus background, gentle natural light",
        "Rich jewel tones, regal velvet-style background, classical portrait lighting",
    ],
    "faceless": [
        "Warm terracotta and gold leaf accents, abstract expressionist background",
        "Cobalt blue and clean white, minimalist contemporary style",
        "Muted earthy tones, pops of ochre, gestural brushwork",
        "Deep orange and coral sunset background, silhouette focus",
        "Blush and forest green, floral abstract elements",
        "Charcoal gestural lines on cream, minimalist figure study",
        "Deep purple and turquoise colour washes, expressive portrait style",
    ],
}


def build_custom_prompt(
    portrait_type: str,
    style_variation: str,
    dry_run: bool = False,
) -> str:
    if portrait_type not in PORTRAIT_TYPES:
        raise ValueError(f"Unknown portrait type '{portrait_type}'. Use: {list(PORTRAIT_TYPES.keys())}")

    pt = PORTRAIT_TYPES[portrait_type]

    if dry_run:
        logger.info(f"[DRY RUN] Returning mock custom {portrait_type} portrait prompt")
        subject = "the pet" if portrait_type == "pet" else "a faceless figure"
        return (
            f"A stunning oil on canvas portrait of {subject} from the reference photo, "
            f"painted in an expressive impasto style with {style_variation.lower()}. "
            f"Thick palette knife marks and visible brushstrokes give three-dimensional texture. "
            f"Soft diffused studio lighting, centred composition with the subject as the clear focal point. "
            f"Contemporary fine art portrait tradition, portrait 2:3 format. "
            f"Photorealistic painting detail, ultra high resolution, 300 DPI print quality, museum quality fine art."
        )

    config = load_config()
    api_key = get_env("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    user_message = (
        f"Portrait type: {pt['label']}\n"
        f"Style and colour variation: {style_variation}\n\n"
        f"Generate the image generation prompt."
    )

    logger.info(f"Custom portrait: Generating prompt for {pt['label']}")
    response = client.messages.create(
        model=config.get("claude_model", "claude-sonnet-4-6"),
        max_tokens=300,
        system=pt["system_prompt"],
        messages=[{"role": "user", "content": user_message}],
    )
    prompt = response.content[0].text.strip()
    logger.info(f"Custom portrait: Prompt generated ({len(prompt.split())} words)")
    return prompt


def get_next_variation(portrait_type: str, state: dict) -> str:
    variations = STYLE_VARIATIONS.get(portrait_type, STYLE_VARIATIONS["pet"])
    idx = state.get(f"{portrait_type}_variation_index", 0) % len(variations)
    return variations[idx]


def advance_variation_state(portrait_type: str, state: dict) -> dict:
    variations = STYLE_VARIATIONS.get(portrait_type, STYLE_VARIATIONS["pet"])
    key = f"{portrait_type}_variation_index"
    state[key] = (state.get(key, 0) + 1) % len(variations)
    return state
