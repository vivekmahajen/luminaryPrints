import io
import os
import time
import requests
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw
from utils.config import get_env
from utils.logger import get_logger

logger = get_logger(__name__)

FAL_TEXT2IMG_URL = "https://queue.fal.run/fal-ai/flux-pro/v1.1"
POLL_INTERVAL = 3
MAX_POLLS = 60


def upload_image_to_fal(image_path: str | Path) -> str:
    """Upload a local image to fal.ai storage using fal-client."""
    api_key = get_env("FAL_API_KEY")
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    logger.info(f"fal.ai: Uploading reference image {image_path.name} ({image_path.stat().st_size // 1024} KB)")

    os.environ["FAL_KEY"] = api_key
    import fal_client
    url = fal_client.upload_file(str(image_path))

    if not url:
        raise RuntimeError("fal_client.upload_file returned no URL")

    logger.info(f"fal.ai: Reference image uploaded — {url}")
    return url


def _generate_abstract_background(prompt: str) -> Image.Image:
    """Generate a pure abstract painting background using Flux Pro text-to-image."""
    api_key = get_env("FAL_API_KEY")
    headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}

    payload = {
        "prompt": prompt,
        "image_size": "portrait_4_3",
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "num_images": 1,
        "enable_safety_checker": True,
        "output_format": "jpeg",
    }

    logger.info("fal.ai: Generating abstract background")
    resp = requests.post(FAL_TEXT2IMG_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    job = resp.json()
    request_id = job.get("request_id")
    status_url = job.get("status_url") or f"https://queue.fal.run/fal-ai/flux-pro/requests/{request_id}/status"
    result_url = job.get("response_url") or f"https://queue.fal.run/fal-ai/flux-pro/requests/{request_id}"

    for poll in range(MAX_POLLS):
        time.sleep(POLL_INTERVAL)
        status = requests.get(status_url, headers=headers, timeout=15).json().get("status", "")
        logger.info(f"fal.ai bg: Poll {poll + 1} — {status}")
        if status == "COMPLETED":
            result = requests.get(result_url, headers=headers, timeout=30).json()
            image_url = result["images"][0]["url"]
            img_bytes = requests.get(image_url, timeout=60).content
            logger.info(f"fal.ai: Background generated ({len(img_bytes) // 1024} KB)")
            return Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        if status in ("FAILED", "ERROR"):
            raise RuntimeError("fal.ai: Background generation failed")

    raise TimeoutError("fal.ai: Background generation timed out")


def _composite_subject_onto_background(
    subject_path: str | Path,
    background: Image.Image,
) -> bytes:
    """
    Place the subject photo in the center of the abstract background
    with a soft feathered oval mask so it blends naturally.
    """
    bg_w, bg_h = background.size

    # Load and resize subject to fill ~65% of background height, centred
    subject = Image.open(subject_path).convert("RGBA")
    target_h = int(bg_h * 0.70)
    ratio = target_h / subject.height
    target_w = int(subject.width * ratio)
    subject = subject.resize((target_w, target_h), Image.LANCZOS)

    # Create oval mask with feathered edge
    mask = Image.new("L", (target_w, target_h), 0)
    draw = ImageDraw.Draw(mask)
    pad = int(min(target_w, target_h) * 0.05)
    draw.ellipse([pad, pad, target_w - pad, target_h - pad], fill=255)
    # Feather/blur the mask edge
    feather = max(int(min(target_w, target_h) * 0.08), 10)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    subject.putalpha(mask)

    # Paste centred onto background
    paste_x = (bg_w - target_w) // 2
    paste_y = (bg_h - target_h) // 2
    composite = background.copy()
    composite.paste(subject, (paste_x, paste_y), subject)

    # Convert to JPEG bytes
    out = io.BytesIO()
    composite.convert("RGB").save(out, format="JPEG", quality=95)
    logger.info(f"Composite: subject {target_w}x{target_h} centred on {bg_w}x{bg_h} background")
    return out.getvalue()


def generate_custom_portrait(prompt: str, reference_image_url: str, reference_local_path: str = "", strength: float = 0.85) -> bytes:
    """
    Two-step composite approach:
    1. Generate abstract background with Flux Pro (text-to-image)
    2. Composite the reference pet/person photo centred on the background
       with a soft feathered oval mask

    This guarantees the abstract background is always visible and the
    subject is always clearly centred, regardless of model behaviour.
    """
    background = _generate_abstract_background(prompt)

    if reference_local_path and Path(reference_local_path).exists():
        logger.info("Compositing reference photo onto abstract background")
        return _composite_subject_onto_background(reference_local_path, background)

    # Fallback: download reference from URL and composite
    logger.info("Downloading reference image for compositing")
    img_bytes = requests.get(reference_image_url, timeout=60).content
    tmp = io.BytesIO(img_bytes)
    subject = Image.open(tmp).convert("RGBA")
    tmp_subject = io.BytesIO()
    subject.save(tmp_subject, format="PNG")
    tmp_subject.seek(0)

    with Path("_tmp_ref.png").open("wb") as f:
        f.write(tmp_subject.getvalue())
    result = _composite_subject_onto_background("_tmp_ref.png", background)
    Path("_tmp_ref.png").unlink(missing_ok=True)
    return result
