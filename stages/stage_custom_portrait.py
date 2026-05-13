import time
import anthropic
from pathlib import Path
from utils.config import load_config, get_env
from utils.logger import get_logger

logger = get_logger(__name__)

# img2img strength: lower = more faithful to reference photo.
# 0.60 keeps the subject's face/body highly recognisable while the
# background is transformed into the abstract style.
PORTRAIT_STRENGTH = 0.60

PORTRAIT_TYPES = {
    "pet": {
        "label": "Custom Pet Portrait",
        "style_id": "custom_pet_portrait",
        "system_prompt": """You are an expert AI art director specialising in premium 3D pop-out pet portrait prints.

The effect you must describe:
- The pet from the reference photo appears in photorealistic 3D detail, centered in the image
- The pet's face, fur texture, and body are rendered with hyperrealistic depth and clarity — they appear to physically pop out of the canvas
- The surrounding background is a bold abstract painting: thick impasto brushstrokes, swirling palette knife marks, or vibrant colour field
- The contrast between the realistic subject and painterly background is the defining visual feature
- The pet occupies the central 60% of the image, clearly visible from head to mid-body or full body

Prompt rules:
1. Open with: "Hyperrealistic 3D portrait of the pet from the reference photo, centered and emerging from the canvas with striking depth and dimension"
2. Describe the abstract background style using the provided colour variation
3. Specify the lighting: dramatic rim light, golden studio light, or soft diffused light that makes the subject pop
4. The subject must be clearly identifiable — full face, expressive eyes, visible body
5. End with: "ultra high resolution, 300 DPI print quality, museum quality fine art, photorealistic subject against painterly abstract background"
6. 90-120 words, single flowing paragraph, portrait 2:3 format

Return ONLY the prompt text.""",
    },
    "faceless": {
        "label": "Custom Faceless Portrait",
        "style_id": "custom_faceless_portrait",
        "system_prompt": """You are an expert AI art director specialising in premium 3D pop-out contemporary portrait prints.

The effect you must describe:
- The person from the reference photo appears in photorealistic 3D detail, centered in the image
- The body, clothing, hair, and posture are rendered with hyperrealistic depth — the figure appears to physically emerge from the canvas
- The face is present and clearly visible but stylised with expressive painterly treatment rather than exact facial likeness
- The surrounding background is a bold abstract painting: gestural brushstrokes, colour washes, or impasto texture
- The contrast between the 3D figure and painterly background is the defining visual feature
- The figure occupies the central 60% of the image, clearly showing upper body or full body

Prompt rules:
1. Open with: "Hyperrealistic 3D contemporary portrait of the figure from the reference photo, centered and emerging from the canvas with striking depth and dimension"
2. Describe the abstract background style using the provided colour variation
3. Specify dramatic lighting that creates depth: rim lighting, golden hour side light, or studio spotlight
4. Face and body should be clearly visible — expressive, stylised but present
5. End with: "ultra high resolution, 300 DPI print quality, museum quality fine art, photorealistic figure against painterly abstract background"
6. 90-120 words, single flowing paragraph, portrait 2:3 format

Return ONLY the prompt text.""",
    },
}

STYLE_VARIATIONS = {
    "pet": [
        "Swirling gold and ivory impasto abstract background, warm studio spotlight",
        "Bold cobalt blue and burnt orange abstract expressionist background, dramatic rim light",
        "Soft blush and rose palette knife abstract background, gentle diffused light",
        "Deep emerald and bronze gestural abstract background, golden hour side light",
        "Charcoal and silver textured abstract background, sharp studio spotlight",
        "Vibrant teal and coral colour field abstract background, dramatic backlight",
        "Warm champagne and sage palette knife abstract background, soft natural window light",
    ],
    "faceless": [
        "Warm terracotta and gold leaf abstract expressionist background, golden studio spotlight",
        "Cobalt blue and pure white gestural abstract background, sharp rim lighting",
        "Muted ochre and earthy brown impasto abstract background, warm side light",
        "Deep orange and coral colour wash abstract background, dramatic backlight halo",
        "Blush pink and forest green botanical abstract background, soft diffused studio light",
        "Charcoal and cream gestural abstract background, single overhead spotlight",
        "Deep purple and turquoise swirling abstract background, cool dramatic rim light",
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
        subject = "the pet" if portrait_type == "pet" else "the figure"
        return (
            f"Hyperrealistic 3D portrait of {subject} from the reference photo, centered and emerging "
            f"from the canvas with striking depth and dimension. "
            f"The subject's face and body are rendered with photorealistic detail — fur texture, eyes, "
            f"and form pop out with three-dimensional clarity. "
            f"Surrounding background: {style_variation.lower()}, thick palette knife marks and swirling "
            f"impasto brushstrokes creating dramatic contrast with the realistic subject. "
            f"Dramatic rim lighting sculpts the subject against the abstract background. "
            f"Centered composition, portrait 2:3 format. "
            f"Ultra high resolution, 300 DPI print quality, museum quality fine art, "
            f"photorealistic subject against painterly abstract background."
        )

    config = load_config()
    api_key = get_env("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    user_message = (
        f"Portrait type: {pt['label']}\n"
        f"Background style and colour variation: {style_variation}\n\n"
        f"Generate the image generation prompt for this 3D pop-out portrait."
    )

    logger.info(f"Custom portrait: Generating 3D pop-out prompt for {pt['label']}")
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
