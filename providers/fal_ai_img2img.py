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
    """Generate abstract background using fal-ai/flux-pro/v1.1 (proven working)."""
    api_key = get_env("FAL_API_KEY")
    headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}

    bg_prompt = (
        f"Abstract oil painting. {prompt} "
        f"Bold impasto brushstrokes, thick palette knife marks, rich saturated colours, "
        f"dramatic painterly texture. No people, no animals, no faces. Pure abstract art."
    )
    payload = {
        "prompt": bg_prompt,
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
    request_id = job["request_id"]
    status_url = job.get("status_url") or f"https://queue.fal.run/fal-ai/flux-pro/requests/{request_id}/status"
    result_url = job.get("response_url") or f"https://queue.fal.run/fal-ai/flux-pro/requests/{request_id}"
    logger.info(f"fal.ai bg: request_id={request_id}")

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
            bg = bg.resize((width, height), Image.LANCZOS)
            logger.info(f"fal.ai bg: Generated and resized to {width}x{height}")
            return bg
        if status in ("FAILED", "ERROR"):
            raise RuntimeError("fal.ai: Background generation failed")

    raise TimeoutError("fal.ai: Background generation timed out")


def _abstract_overlay_portrait(reference_path: str | Path, abstract: Image.Image) -> bytes:
    """
    Blend the abstract painting ON TOP of the pet photo.

    The abstract defines the colours and texture; the pet's face and
    body are visible through it — like paint applied over a photograph.

    Blend profile:
      - Edges/background: abstract at 90% (strong abstract)
      - Mid zone:         abstract at 72%
      - Face centre:      abstract at 50% (face clearly visible through paint)

    This creates the "abstract shaped by the animal" look where the
    brushstrokes define the form rather than a photo sitting on a background.
    """
    with Image.open(reference_path) as ref:
        pet = ref.convert("RGB").resize(abstract.size, Image.LANCZOS)

    w, h = pet.size

    # Radial vignette: white=face centre (more pet), black=edges (more abstract)
    vignette = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(vignette)
    cx, cy = int(w * 0.50), int(h * 0.38)   # slightly above centre — face area
    oval_w, oval_h = int(w * 0.55), int(h * 0.50)
    draw.ellipse([cx - oval_w//2, cy - oval_h//2, cx + oval_w//2, cy + oval_h//2], fill=255)
    blur_r = max(int(min(w, h) * 0.30), 40)
    vignette = vignette.filter(ImageFilter.GaussianBlur(radius=blur_r))

    # Two blend levels
    face_blend  = Image.blend(pet, abstract, alpha=0.50)  # 50% abstract, face visible
    outer_blend = Image.blend(pet, abstract, alpha=0.90)  # 90% abstract, heavily painted

    # Composite: where vignette=255 → face_blend; where vignette=0 → outer_blend
    result = Image.composite(face_blend, outer_blend, vignette)

    out = io.BytesIO()
    result.convert("RGB").save(out, format="JPEG", quality=95)
    logger.info(f"Abstract overlay: {w}x{h}, face at ({cx},{cy}), blur={blur_r}px")
    return out.getvalue()


def generate_custom_portrait(
    prompt: str,
    reference_image_url: str,
    reference_local_path: str = "",
    strength: float = 0.85,
) -> bytes:
    """
    Abstract overlay portrait:
    1. Generate bold abstract painting with fal-ai/flux-pro/v1.1
    2. Blend it ON TOP of the pet photo — abstract defines the colours
       and texture, pet face/body visible through the paint layers
    """
    local_path = reference_local_path if reference_local_path else None
    if not local_path or not Path(local_path).exists():
        logger.info("Downloading reference image")
        img_bytes = requests.get(reference_image_url, timeout=60).content
        local_path = "_tmp_portrait_ref.jpg"
        Path(local_path).write_bytes(img_bytes)

    with Image.open(local_path) as ref:
        ref_w, ref_h = ref.size

    abstract = _generate_abstract_background(prompt, ref_w, ref_h)
    result_bytes = _abstract_overlay_portrait(local_path, abstract)

    if reference_local_path == "" and Path("_tmp_portrait_ref.jpg").exists():
        Path("_tmp_portrait_ref.jpg").unlink(missing_ok=True)

    logger.info(f"Portrait complete: {len(result_bytes) // 1024} KB")
    return result_bytes
