import io
import os
import time
import requests
from pathlib import Path
from PIL import Image
from utils.config import get_env
from utils.logger import get_logger

logger = get_logger(__name__)

FAL_IMG2IMG_MODEL = "fal-ai/flux/dev/image-to-image"
FAL_TEXT2IMG_URL = "https://queue.fal.run/fal-ai/flux-pro/v1.1"
POLL_INTERVAL = 3
MAX_POLLS = 80


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


def _poll_job(status_url: str, result_url: str, headers: dict, label: str) -> dict:
    """Poll a fal.ai queue job until COMPLETED, return result JSON."""
    for poll in range(MAX_POLLS):
        time.sleep(POLL_INTERVAL)
        st = requests.get(status_url, headers=headers, timeout=15)
        st.raise_for_status()
        body = st.json()
        status = body.get("status", "")
        inference = body.get("metrics", {}).get("inference_time", 0)
        logger.info(f"fal.ai {label}: Poll {poll + 1} — {status} ({inference:.1f}s)")
        if status == "COMPLETED":
            result = requests.get(result_url, headers=headers, timeout=30)
            result.raise_for_status()
            return result.json()
        if status in ("FAILED", "ERROR"):
            raise RuntimeError(f"fal.ai {label}: Job failed — {body}")
    raise TimeoutError(f"fal.ai {label}: Timed out after {MAX_POLLS * POLL_INTERVAL}s")


def generate_custom_portrait(
    prompt: str,
    reference_image_url: str,
    reference_local_path: str = "",
    strength: float = 0.85,
) -> bytes:
    """
    Image-to-image portrait transformation:

    Submits the reference photo to fal-ai/flux/dev/image-to-image at
    strength=0.75. The model keeps the pet's structure and pose but
    renders the entire image — pet AND background — in an abstract
    painterly style. The result looks like an abstract painting where
    the pet's form emerges from the brushstrokes, not a photo pasted
    on a background.

    Prompt focuses on the abstract/painterly treatment so the AI
    applies that style throughout the image rather than just the edges.
    """
    api_key = get_env("FAL_API_KEY")
    headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}
    os.environ["FAL_KEY"] = api_key
    import fal_client

    # Ensure we have a local copy for upload
    local_path = reference_local_path if reference_local_path else None
    if not local_path or not Path(local_path).exists():
        logger.info("Downloading reference image")
        img_bytes = requests.get(reference_image_url, timeout=60).content
        local_path = "_tmp_portrait_ref.jpg"
        Path(local_path).write_bytes(img_bytes)

    # Upload reference photo to fal.ai storage
    ref_url = fal_client.upload_file(str(local_path))
    logger.info(f"Reference uploaded — {ref_url}")

    # Prompt: style guides the transformation of the whole image
    styled_prompt = (
        f"{prompt} "
        f"Abstract expressionist oil painting. The entire image is rendered as bold "
        f"impasto brushstrokes and thick palette knife marks. The pet's face and body "
        f"emerge dramatically from swirling abstract colour fields — the painterly "
        f"strokes define the fur, eyes and form. Rich saturated colours. "
        f"The pet appears to push forward from the canvas in 3D. "
        f"Museum quality fine art print, 300 DPI."
    )

    payload = {
        "image_url": ref_url,
        "prompt": styled_prompt,
        "strength": 0.75,
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "num_images": 1,
        "enable_safety_checker": True,
        "output_format": "jpeg",
    }

    logger.info("fal.ai img2img: Submitting job")
    start = time.time()
    resp = requests.post(
        f"https://queue.fal.run/{FAL_IMG2IMG_MODEL}",
        json=payload,
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    job = resp.json()
    request_id = job["request_id"]
    # fal-ai/flux app_id → result URL uses fal-ai/flux (without /dev/image-to-image)
    status_url = job.get("status_url") or f"https://queue.fal.run/fal-ai/flux/requests/{request_id}/status"
    result_url = job.get("response_url") or f"https://queue.fal.run/fal-ai/flux/requests/{request_id}"
    logger.info(f"fal.ai img2img: request_id={request_id}")
    logger.info(f"fal.ai img2img: status_url={status_url}")
    logger.info(f"fal.ai img2img: result_url={result_url}")

    result = _poll_job(status_url, result_url, headers, "img2img")

    images = result.get("images") or []
    if not images:
        raise RuntimeError(f"fal.ai img2img: No images in response: {result}")

    image_url = images[0].get("url") if isinstance(images[0], dict) else images[0]
    logger.info(f"fal.ai img2img: Downloading result from {image_url}")
    img = requests.get(image_url, timeout=60)
    img.raise_for_status()

    elapsed = time.time() - start
    logger.info(f"fal.ai img2img: Done in {elapsed:.1f}s, {len(img.content) // 1024} KB")

    if reference_local_path == "" and Path("_tmp_portrait_ref.jpg").exists():
        Path("_tmp_portrait_ref.jpg").unlink(missing_ok=True)

    return img.content
