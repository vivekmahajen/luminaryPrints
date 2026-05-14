import io
import os
import time
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
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

    logger.info(f"fal.ai: Uploading {image_path.name} ({image_path.stat().st_size // 1024} KB)")
    os.environ["FAL_KEY"] = api_key
    import fal_client
    url = fal_client.upload_file(str(image_path))
    if not url:
        raise RuntimeError("fal_client.upload_file returned no URL")
    logger.info(f"fal.ai: Uploaded — {url}")
    return url


def _generate_abstract_background(prompt: str, width: int, height: int) -> Image.Image:
    """
    Generate a pure abstract painting background using fal-ai/flux-pro/v1.1
    (the same proven-working text-to-image endpoint used by the daily pipeline).
    """
    api_key = get_env("FAL_API_KEY")
    headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}

    # Choose the closest fal.ai portrait size
    if height > width:
        image_size = "portrait_4_3"
    else:
        image_size = "landscape_4_3"

    bg_prompt = (
        f"Abstract oil painting for a 3D pop-out art print. {prompt} "
        f"Bold impasto brushstrokes, thick palette knife marks, rich saturated colour fields, "
        f"dramatic painterly texture. No people, no animals, no faces. Pure abstract art only."
    )

    payload = {
        "prompt": bg_prompt,
        "image_size": image_size,
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
    logger.info(f"fal.ai: Background job submitted, request_id={request_id}")

    for poll in range(MAX_POLLS):
        time.sleep(POLL_INTERVAL)
        st = requests.get(status_url, headers=headers, timeout=15)
        st.raise_for_status()
        status = st.json().get("status", "")
        logger.info(f"fal.ai bg: Poll {poll + 1} — {status}")
        if status == "COMPLETED":
            result = requests.get(result_url, headers=headers, timeout=30).json()
            image_url = result["images"][0]["url"]
            img_bytes = requests.get(image_url, timeout=60).content
            bg = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            logger.info(f"fal.ai bg: Generated {bg.width}x{bg.height}")
            return bg
        if status in ("FAILED", "ERROR"):
            raise RuntimeError("fal.ai: Background generation failed")

    raise TimeoutError("fal.ai: Background generation timed out")


def _sample_border_color(img: Image.Image) -> tuple[int, int, int]:
    """Sample the average colour from the border strip of an image."""
    w, h = img.size
    border = 40
    strips = [
        img.crop((0, 0, w, border)),
        img.crop((0, h - border, w, h)),
        img.crop((0, 0, border, h)),
        img.crop((w - border, 0, w, h)),
    ]
    r = g = b = count = 0
    for strip in strips:
        small = strip.resize((8, 8), Image.LANCZOS).convert("RGB")
        for px in small.getdata():
            r += px[0]; g += px[1]; b += px[2]; count += 1
    return (r // count, g // count, b // count)


def _composite_portrait(reference_path: str | Path, background: Image.Image) -> bytes:
    """
    Composite the reference photo onto the abstract background.

    Subject occupies 68% of the frame so the abstract background is
    clearly visible around it. A 25% Gaussian-feathered oval mask gives
    a very gradual edge. A colour wash sampled from the background is
    blended into the subject's edge zone so the colours integrate rather
    than look like a hard paste.
    """
    bg_w, bg_h = background.size

    # Scale subject to 68% of frame — leaves plenty of background visible
    with Image.open(reference_path) as ref:
        ref = ref.convert("RGB")
    scale = min(bg_w / ref.width, bg_h / ref.height) * 0.68
    new_w = int(ref.width * scale)
    new_h = int(ref.height * scale)
    ref = ref.resize((new_w, new_h), Image.LANCZOS)

    paste_x = (bg_w - new_w) // 2
    paste_y = (bg_h - new_h) // 2

    # --- Build the main alpha mask (white centre, black edges) ---
    mask = Image.new("L", (new_w, new_h), 0)
    draw = ImageDraw.Draw(mask)
    inner_w = int(new_w * 0.68)
    inner_h = int(new_h * 0.80)
    ix = (new_w - inner_w) // 2
    iy = (new_h - inner_h) // 2
    draw.ellipse([ix, iy, ix + inner_w, iy + inner_h], fill=255)
    feather = max(int(min(new_w, new_h) * 0.25), 30)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))

    # --- Colour-wash: bleed background palette into the subject edges ---
    bg_color = _sample_border_color(background)
    color_wash = Image.new("RGB", (new_w, new_h), bg_color)

    # Edge-wash mask: strong where the alpha is low (edges), absent at centre
    wash_mask = Image.new("L", (new_w, new_h), 0)
    draw2 = ImageDraw.Draw(wash_mask)
    draw2.ellipse([ix, iy, ix + inner_w, iy + inner_h], fill=255)
    wash_mask = wash_mask.filter(ImageFilter.GaussianBlur(radius=feather))
    # Invert and soften: 0 = centre (no wash), 180 = edges (strong wash)
    wash_mask = wash_mask.point(lambda p: max(0, 180 - int(p * 0.85)))

    ref_rgba = ref.copy().convert("RGBA")
    wash_rgba = color_wash.convert("RGBA")
    wash_rgba.putalpha(wash_mask)
    # Blend colour wash onto subject
    ref_with_wash = Image.alpha_composite(ref_rgba, wash_rgba).convert("RGB")

    # --- Final composite onto background ---
    ref_final = ref_with_wash.convert("RGBA")
    ref_final.putalpha(mask)

    canvas = background.copy().convert("RGBA")
    canvas.paste(ref_final, (paste_x, paste_y), ref_final)

    out = io.BytesIO()
    canvas.convert("RGB").save(out, format="JPEG", quality=95)
    logger.info(
        f"Composite: subject {new_w}x{new_h} at ({paste_x},{paste_y}) "
        f"on {bg_w}x{bg_h}, feather={feather}px, wash_colour={bg_color}"
    )
    return out.getvalue()


def generate_custom_portrait(
    prompt: str,
    reference_image_url: str,
    reference_local_path: str = "",
    strength: float = 0.85,
) -> bytes:
    """
    Two-step approach (100% reliable — no broken inpainting endpoints):

    1. Generate a bold abstract background with fal-ai/flux-pro/v1.1
       (the same proven endpoint used by the daily print pipeline)
    2. Composite the reference photo onto the background using a wide
       Gaussian-feathered radial mask so the subject blends naturally
       into the painted background — photorealistic subject, abstract surround

    The wide feather (22% of frame) prevents the "cutout" look.
    """
    local_path = reference_local_path if reference_local_path else None

    if not local_path or not Path(local_path).exists():
        logger.info("Downloading reference image")
        img_bytes = requests.get(reference_image_url, timeout=60).content
        local_path = "_tmp_portrait_ref.jpg"
        Path(local_path).write_bytes(img_bytes)

    # Read reference dimensions to size the background correctly
    with Image.open(local_path) as ref:
        ref_w, ref_h = ref.size

    logger.info(f"Reference image: {ref_w}x{ref_h}")

    # Step 1: Generate abstract background
    background = _generate_abstract_background(prompt, ref_w, ref_h)

    # Resize background to match reference dimensions for clean compositing
    background = background.resize((ref_w, ref_h), Image.LANCZOS)

    # Step 2: Composite subject onto background
    result_bytes = _composite_portrait(local_path, background)

    if reference_local_path == "" and Path("_tmp_portrait_ref.jpg").exists():
        Path("_tmp_portrait_ref.jpg").unlink(missing_ok=True)

    logger.info(f"Custom portrait complete: {len(result_bytes) // 1024} KB")
    return result_bytes
