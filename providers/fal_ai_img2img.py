import io
import os
import time
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
from utils.config import get_env
from utils.logger import get_logger

logger = get_logger(__name__)

FAL_MODEL = "fal-ai/flux-pro/v1.1/fill"


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


def _make_background_mask(image_path: str | Path) -> bytes:
    """
    Create an inpainting mask where:
      WHITE = repaint as abstract (background area)
      BLACK = keep exactly as-is (subject in center)

    The subject area is a soft oval covering ~65% of the image height
    centered in the frame. The feathered edge creates a natural blend
    between the original subject and the AI-generated abstract background.
    """
    with Image.open(image_path) as img:
        w, h = img.size

    # Start with all-white mask (repaint everything)
    mask = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(mask)

    # Draw a black oval in the center (= keep subject)
    subject_w = int(w * 0.72)
    subject_h = int(h * 0.78)
    x0 = (w - subject_w) // 2
    y0 = (h - subject_h) // 2
    x1 = x0 + subject_w
    y1 = y0 + subject_h
    draw.ellipse([x0, y0, x1, y1], fill=0)

    # Heavily blur the mask edge so subject blends into the painted background
    feather = max(int(min(w, h) * 0.10), 20)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))

    buf = io.BytesIO()
    mask.convert("RGB").save(buf, format="PNG")
    logger.info(f"Mask created: {w}x{h}, subject oval {subject_w}x{subject_h}, feather={feather}px")
    return buf.getvalue()


def generate_custom_portrait(
    prompt: str,
    reference_image_url: str,
    reference_local_path: str = "",
    strength: float = 0.85,
) -> bytes:
    """
    Inpainting approach — best quality for subject + abstract background:

    1. Create a mask: WHITE = background (repaint as abstract),
                      BLACK = subject center (keep original photo)
    2. Send image + mask + abstract prompt to Flux Pro Fill (inpainting)
       via fal_client.subscribe() which handles queue polling correctly
    3. Flux repaints only the background as abstract art while
       preserving the subject pixel-perfect from the original photo

    Result: natural seamless blend — no cutout look, no compositing artifacts.
    """
    api_key = get_env("FAL_API_KEY")
    os.environ["FAL_KEY"] = api_key
    import fal_client

    local_path = reference_local_path if reference_local_path else None

    if not local_path or not Path(local_path).exists():
        logger.info("Downloading reference image for mask creation")
        img_bytes = requests.get(reference_image_url, timeout=60).content
        local_path = "_tmp_portrait_ref.jpg"
        Path(local_path).write_bytes(img_bytes)

    # Build and upload the mask
    logger.info("Creating background mask")
    mask_bytes = _make_background_mask(local_path)
    mask_tmp = "_tmp_portrait_mask.png"
    Path(mask_tmp).write_bytes(mask_bytes)
    mask_url = fal_client.upload_file(mask_tmp)
    Path(mask_tmp).unlink(missing_ok=True)
    logger.info(f"Mask uploaded — {mask_url}")

    # Inpainting: repaint background only, keep subject
    abstract_prompt = (
        f"{prompt} The background surrounding the subject is a bold abstract oil painting "
        f"with thick impasto brushstrokes, swirling palette knife marks, and rich painterly texture. "
        f"The subject in the center remains photorealistic and sharp."
    )

    arguments = {
        "image_url": reference_image_url,
        "mask_url": mask_url,
        "prompt": abstract_prompt,
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "num_images": 1,
        "enable_safety_checker": True,
        "output_format": "jpeg",
    }

    logger.info("fal.ai inpaint: Submitting job via fal_client.subscribe()")
    start = time.time()

    def _on_update(update):
        logger.info(f"fal.ai inpaint: {type(update).__name__}")

    result = fal_client.subscribe(
        FAL_MODEL,
        arguments=arguments,
        with_logs=False,
        on_queue_update=_on_update,
    )
    images = result.get("images", [])
    if not images:
        raise RuntimeError("fal.ai inpaint: No images in response")

    image_url = images[0].get("url")
    logger.info(f"fal.ai inpaint: Downloading result from {image_url}")
    img = requests.get(image_url, timeout=60)
    img.raise_for_status()

    elapsed = time.time() - start
    logger.info(f"fal.ai inpaint: Done in {elapsed:.1f}s, {len(img.content) // 1024} KB")

    # Clean up temp files
    if reference_local_path == "" and Path("_tmp_portrait_ref.jpg").exists():
        Path("_tmp_portrait_ref.jpg").unlink(missing_ok=True)

    return img.content
