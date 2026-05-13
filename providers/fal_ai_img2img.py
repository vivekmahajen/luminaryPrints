import os
import time
import requests
from pathlib import Path
from utils.config import get_env
from utils.logger import get_logger

logger = get_logger(__name__)

FAL_REDUX_URL = "https://queue.fal.run/fal-ai/flux-pro/v1.1/redux"
POLL_INTERVAL = 3
MAX_POLLS = 60


def upload_image_to_fal(image_path: str | Path) -> str:
    """Upload a local image to fal.ai storage using the official fal-client and return its URL."""
    api_key = get_env("FAL_API_KEY")
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    logger.info(f"fal.ai: Uploading reference image {image_path.name} ({image_path.stat().st_size // 1024} KB)")

    # fal-client reads FAL_KEY from environment
    os.environ["FAL_KEY"] = api_key

    import fal_client
    url = fal_client.upload_file(str(image_path))

    if not url:
        raise RuntimeError("fal_client.upload_file returned no URL")

    logger.info(f"fal.ai: Reference image uploaded — {url}")
    return url


def generate_custom_portrait(prompt: str, reference_image_url: str, strength: float = 0.80) -> bytes:
    """
    Generate an image using fal.ai Flux Redux (image-to-image).
    strength: 0.0 = identical to reference, 1.0 = completely new image.
    0.75-0.85 keeps subject recognisable while applying the artistic style.
    """
    api_key = get_env("FAL_API_KEY")
    headers = {
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "image_url": reference_image_url,
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "strength": strength,
        "num_images": 1,
        "enable_safety_checker": True,
        "output_format": "jpeg",
    }

    logger.info(f"fal.ai redux: Submitting image-to-image request (strength={strength})")
    start = time.time()

    resp = requests.post(FAL_REDUX_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    job = resp.json()
    request_id = job.get("request_id")
    status_url = f"https://queue.fal.run/fal-ai/flux-pro/requests/{request_id}/status"
    result_url = f"https://queue.fal.run/fal-ai/flux-pro/requests/{request_id}"

    logger.info(f"fal.ai redux: Job submitted, request_id={request_id}")

    for poll in range(MAX_POLLS):
        time.sleep(POLL_INTERVAL)
        status_resp = requests.get(status_url, headers=headers, timeout=15)
        status_resp.raise_for_status()
        status = status_resp.json().get("status", "")
        logger.info(f"fal.ai redux: Poll {poll + 1}/{MAX_POLLS} — status={status}")

        if status == "COMPLETED":
            result = requests.get(result_url, headers=headers, timeout=30).json()
            images = result.get("images", [])
            if not images:
                raise RuntimeError("fal.ai redux: No images in response")
            image_url = images[0].get("url")
            logger.info(f"fal.ai redux: Downloading from {image_url}")
            img = requests.get(image_url, timeout=60)
            img.raise_for_status()
            elapsed = time.time() - start
            logger.info(f"fal.ai redux: Done in {elapsed:.1f}s, {len(img.content) / 1024:.1f} KB")
            return img.content

        if status in ("FAILED", "ERROR"):
            raise RuntimeError(f"fal.ai redux: Job failed — status={status}")

    raise TimeoutError(f"fal.ai redux: Timed out after {MAX_POLLS * POLL_INTERVAL}s")
