import time
import requests
from utils.logger import get_logger
from utils.config import get_env

logger = get_logger(__name__)

FAL_QUEUE_URL = "https://queue.fal.run/fal-ai/flux-pro/v1.1"
POLL_INTERVAL = 3
MAX_POLLS = 60  # 3 minutes


def generate(prompt: str) -> bytes:
    api_key = get_env("FAL_API_KEY")
    headers = {
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "image_size": "portrait_4_3",
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "num_images": 1,
        "enable_safety_checker": True,
        "output_format": "jpeg",
    }

    logger.info(f"fal.ai: Submitting generation request to {FAL_QUEUE_URL}")
    start = time.time()
    resp = requests.post(FAL_QUEUE_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    job = resp.json()
    request_id = job.get("request_id")
    status_url = job.get("status_url") or f"https://queue.fal.run/fal-ai/flux-pro/requests/{request_id}/status"
    result_url = job.get("response_url") or f"https://queue.fal.run/fal-ai/flux-pro/requests/{request_id}"

    logger.info(f"fal.ai: Job submitted, request_id={request_id}")

    for poll in range(MAX_POLLS):
        time.sleep(POLL_INTERVAL)
        status_resp = requests.get(status_url, headers=headers, timeout=15)
        status_resp.raise_for_status()
        status_data = status_resp.json()
        status = status_data.get("status", "")
        logger.info(f"fal.ai: Poll {poll + 1}/{MAX_POLLS} — status={status}")

        if status == "COMPLETED":
            result_resp = requests.get(result_url, headers=headers, timeout=30)
            result_resp.raise_for_status()
            result = result_resp.json()
            images = result.get("images", [])
            if not images:
                raise RuntimeError("fal.ai: No images in completed response")
            image_url = images[0].get("url")
            logger.info(f"fal.ai: Downloading image from {image_url}")
            img_resp = requests.get(image_url, timeout=60)
            img_resp.raise_for_status()
            elapsed = time.time() - start
            logger.info(f"fal.ai: Done in {elapsed:.1f}s, {len(img_resp.content) / 1024:.1f} KB")
            return img_resp.content

        if status in ("FAILED", "ERROR"):
            raise RuntimeError(f"fal.ai: Job failed with status={status}")

    raise TimeoutError(f"fal.ai: Timed out after {MAX_POLLS * POLL_INTERVAL}s")
