import io
import os
import time
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
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


def _composite_3d_popout(reference_path: str | Path, background: Image.Image) -> bytes:
    """
    3D pop-out effect: the abstract painting acts as a canvas frame
    and the subject BREAKS THROUGH it from behind, face-first.

    Layering (back to front):
      1. Pet photo scaled to fill the full frame
      2. Abstract painting laid ON TOP with an oval window cut out
         — abstract covers the body/edges, face bursts through the hole
      3. A soft vignette around the window edge deepens the 3D illusion

    The oval window is shifted upward so the face/head is the focal
    point pushing through the painting surface.
    """
    bg_w, bg_h = background.size

    # Scale pet to fill the full frame (abstract covers the edges)
    with Image.open(reference_path) as ref:
        pet = ref.convert("RGB").resize((bg_w, bg_h), Image.LANCZOS)

    # Window mask: WHITE = pet visible (breaks through), BLACK = abstract covers
    # Oval shifted up ~10% so face/head is centred in the window
    win_mask = Image.new("L", (bg_w, bg_h), 0)
    draw = ImageDraw.Draw(win_mask)
    win_w = int(bg_w * 0.52)          # window is 52% of frame width
    win_h = int(bg_h * 0.56)          # window is 56% of frame height
    wx = (bg_w - win_w) // 2
    wy = int(bg_h * 0.10)             # shifted up — face is in top-centre
    draw.ellipse([wx, wy, wx + win_w, wy + win_h], fill=255)

    # Feather the window: soft transition between pet and abstract
    feather = max(int(min(bg_w, bg_h) * 0.13), 25)
    win_mask = win_mask.filter(ImageFilter.GaussianBlur(radius=feather))

    # Composite: where win_mask=255 → pet, where win_mask=0 → abstract
    canvas = Image.composite(pet, background, win_mask)

    out = io.BytesIO()
    canvas.convert("RGB").save(out, format="JPEG", quality=95)
    logger.info(
        f"3D pop-out: pet {bg_w}x{bg_h}, window {win_w}x{win_h} "
        f"at ({wx},{wy}), feather={feather}px"
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

    # Step 2: 3D pop-out composite — abstract on top, pet breaks through
    result_bytes = _composite_3d_popout(local_path, background)

    if reference_local_path == "" and Path("_tmp_portrait_ref.jpg").exists():
        Path("_tmp_portrait_ref.jpg").unlink(missing_ok=True)

    logger.info(f"Custom portrait complete: {len(result_bytes) // 1024} KB")
    return result_bytes
