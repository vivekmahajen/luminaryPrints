"""
Stage: Room Mockup Generator

Reads etsy_listing.json + generation_log.json from an output folder,
generates styled room lifestyle images showing the portrait on a wall.

Outputs per run:
  mockup_bedroom_room.jpg      — AI room scene (fal.ai)
  mockup_living_room_room.jpg  — AI room scene (fal.ai)
  mockup_framed_flatlay.jpg    — PIL framed flat-lay (no API)
"""
import io
import json
import time
import requests
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
from utils.config import get_env
from utils.logger import get_logger

logger = get_logger(__name__)

FAL_TEXT2IMG_URL = "https://queue.fal.run/fal-ai/flux-pro/v1.1"
POLL_INTERVAL = 3
MAX_POLLS = 60

ROOM_SCENES = {
    "bedroom": {
        "interior": (
            "Cozy Scandinavian master bedroom, white plaster wall, warm golden evening light "
            "through sheer curtains, minimalist oak bed frame with linen bedding, "
            "potted fiddle-leaf fig plant in corner, light hardwood floor"
        ),
        "placement": "hanging centered on the main wall above the bed headboard",
    },
    "living-room": {
        "interior": (
            "Modern minimalist living room, light grey walls, concrete floor, "
            "mid-century sofa in warm terracotta, floor lamp with warm glow, "
            "natural afternoon light through large windows, indoor plant"
        ),
        "placement": "displayed prominently on the accent wall beside the sofa",
    },
    "home-office": {
        "interior": (
            "Clean contemporary home office, white wall, light oak desk, "
            "open shelves with books and succulents, warm desk lamp, soft daylight"
        ),
        "placement": "mounted on the wall facing the desk at eye level",
    },
}


def _fal_generate_room(prompt: str) -> Image.Image:
    api_key = get_env("FAL_API_KEY")
    headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}
    payload = {
        "prompt": prompt,
        "image_size": "landscape_4_3",
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "num_images": 1,
        "enable_safety_checker": True,
        "output_format": "jpeg",
    }

    resp = requests.post(FAL_TEXT2IMG_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    job = resp.json()
    rid = job["request_id"]
    status_url = job.get("status_url") or f"https://queue.fal.run/fal-ai/flux-pro/requests/{rid}/status"
    result_url = job.get("response_url") or f"https://queue.fal.run/fal-ai/flux-pro/requests/{rid}"

    for poll in range(MAX_POLLS):
        time.sleep(POLL_INTERVAL)
        st = requests.get(status_url, headers=headers, timeout=15)
        st.raise_for_status()
        status = st.json().get("status", "")
        logger.info(f"fal.ai room: Poll {poll + 1} — {status}")
        if status == "COMPLETED":
            result = requests.get(result_url, headers=headers, timeout=30).json()
            img_bytes = requests.get(result["images"][0]["url"], timeout=60).content
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if status in ("FAILED", "ERROR"):
            raise RuntimeError("fal.ai: Room scene generation failed")

    raise TimeoutError("fal.ai: Room scene generation timed out")


def _build_room_prompt(room_key: str, artwork_desc: str, colour_variation: str) -> str:
    scene = ROOM_SCENES[room_key]
    return (
        f"Professional interior photography. {scene['interior']}. "
        f"A framed fine art canvas print {scene['placement']}: {artwork_desc} "
        f"with {colour_variation.lower()} colour palette, natural oak wood frame with white mat board. "
        f"Soft even lighting illuminates the artwork, photorealistic, high resolution "
        f"interior design photo, magazine quality, no people."
    )


def _make_flat_framed_mockup(portrait_path: Path) -> Image.Image:
    """Portrait + white mat + dark frame + drop shadow on off-white wall background."""
    with Image.open(portrait_path) as img:
        portrait = img.convert("RGB")

    pw, ph = portrait.size
    scale = min(900 / pw, 1100 / ph)
    pw, ph = int(pw * scale), int(ph * scale)
    portrait = portrait.resize((pw, ph), Image.LANCZOS)

    mat_px = int(min(pw, ph) * 0.06)
    frame_px = int(min(pw, ph) * 0.025)
    total_w = pw + 2 * (mat_px + frame_px)
    total_h = ph + 2 * (mat_px + frame_px)

    canvas_w = total_w + 300
    canvas_h = total_h + 300

    # Off-white plaster wall
    canvas = Image.new("RGB", (canvas_w, canvas_h), (242, 238, 230))

    # Drop shadow
    shadow = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    sx = (canvas_w - total_w) // 2 + 10
    sy = (canvas_h - total_h) // 2 + 12
    sdraw.rectangle([sx, sy, sx + total_w, sy + total_h], fill=(0, 0, 0, 90))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=14))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), shadow).convert("RGB")

    cx = (canvas_w - total_w) // 2
    cy = (canvas_h - total_h) // 2

    # Dark walnut frame
    framed = Image.new("RGB", (total_w, total_h), (52, 38, 28))
    # White mat
    mat_img = Image.new("RGB", (pw + 2 * mat_px, ph + 2 * mat_px), (252, 250, 246))
    mat_img.paste(portrait, (mat_px, mat_px))
    framed.paste(mat_img, (frame_px, frame_px))
    canvas.paste(framed, (cx, cy))

    # Subtle edge vignette using numpy
    cy_v, cx_v = canvas_h / 2, canvas_w / 2
    Y, X = np.ogrid[:canvas_h, :canvas_w]
    dist = ((X - cx_v) / canvas_w) ** 2 + ((Y - cy_v) / canvas_h) ** 2
    vig = np.clip(1.0 - 0.35 * (dist / dist.max()), 0.78, 1.0)
    canvas_arr = np.array(canvas, dtype=np.float32) * vig[:, :, None]
    return Image.fromarray(canvas_arr.clip(0, 255).astype(np.uint8))


def generate_mockups(output_folder: Path) -> list[Path]:
    """
    Generate room mockup images for the given output folder.
    Reads etsy_listing.json + generation_log.json, writes mockup_*.jpg files.
    Returns list of paths written.
    """
    etsy_path = output_folder / "etsy_listing.json"
    log_path = output_folder / "generation_log.json"

    if not etsy_path.exists() or not log_path.exists():
        raise FileNotFoundError(f"Missing JSON files in {output_folder}")

    etsy = json.loads(etsy_path.read_text(encoding="utf-8"))
    gen_log = json.loads(log_path.read_text(encoding="utf-8"))

    room_tags = [t.lower().replace("_", "-") for t in etsy.get("room_tags", ["bedroom", "living-room"])]
    colour_variation = gen_log.get("variation", "")
    style = gen_log.get("style", "custom art print")
    artwork_desc = f"{style} — abstract painting"

    # Prefer 8x10 as source for flat-lay (smaller file, faster)
    portrait_path = output_folder / "print_8x10_2400x3000px.jpg"
    if not portrait_path.exists():
        candidates = sorted(output_folder.glob("print_*.jpg"))
        if not candidates:
            raise FileNotFoundError(f"No print_*.jpg in {output_folder}")
        portrait_path = candidates[0]

    written: list[Path] = []

    # AI room scenes
    for room_key in room_tags:
        if room_key not in ROOM_SCENES:
            logger.warning(f"Mockup: unknown room tag '{room_key}', skipping")
            continue

        logger.info(f"Mockup: generating {room_key} room scene via fal.ai")
        prompt = _build_room_prompt(room_key, artwork_desc, colour_variation)

        try:
            scene_img = _fal_generate_room(prompt)
            out_path = output_folder / f"mockup_{room_key.replace('-', '_')}_room.jpg"
            scene_img.save(str(out_path), format="JPEG", quality=92)
            written.append(out_path)
            logger.info(f"Mockup: saved {out_path.name}")
        except Exception as e:
            logger.error(f"Mockup: room scene failed for {room_key} — {e}")

    # PIL flat-lay (always runs, no API)
    try:
        logger.info("Mockup: generating framed flat-lay mockup")
        flatlay = _make_flat_framed_mockup(portrait_path)
        flat_path = output_folder / "mockup_framed_flatlay.jpg"
        flatlay.save(str(flat_path), format="JPEG", quality=92)
        written.append(flat_path)
        logger.info(f"Mockup: saved {flat_path.name}")
    except Exception as e:
        logger.error(f"Mockup: flat-lay failed — {e}")

    logger.info(f"Mockup stage complete — {len(written)} images written to {output_folder.name}")
    return written
