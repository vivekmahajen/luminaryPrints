import time
import requests
from utils.logger import get_logger
from utils.config import get_env

logger = get_logger(__name__)

IDEOGRAM_URL = "https://api.ideogram.ai/generate"


def generate(prompt: str) -> bytes:
    api_key = get_env("IDEOGRAM_API_KEY")
    headers = {
        "Api-Key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "image_request": {
            "prompt": prompt,
            "aspect_ratio": "ASPECT_2_3",
            "model": "V_2",
            "magic_prompt_option": "ON",
        }
    }

    logger.info("Ideogram: Submitting generation request")
    start = time.time()

    resp = requests.post(IDEOGRAM_URL, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    images = data.get("data", [])
    if not images:
        raise RuntimeError("Ideogram: No images in response")

    image_url = images[0].get("url")
    logger.info(f"Ideogram: Downloading image from {image_url}")

    img_resp = requests.get(image_url, timeout=60)
    img_resp.raise_for_status()

    elapsed = time.time() - start
    logger.info(f"Ideogram: Done in {elapsed:.1f}s, {len(img_resp.content) / 1024:.1f} KB")
    return img_resp.content
