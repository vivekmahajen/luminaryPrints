import time
import requests
from openai import OpenAI
from utils.logger import get_logger
from utils.config import get_env

logger = get_logger(__name__)


def generate(prompt: str) -> bytes:
    api_key = get_env("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    logger.info("DALL-E 3: Submitting generation request")
    start = time.time()

    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        size="1024x1792",
        quality="hd",
        style="vivid",
    )

    image_url = response.data[0].url
    logger.info(f"DALL-E 3: Downloading image from {image_url}")

    img_resp = requests.get(image_url, timeout=60)
    img_resp.raise_for_status()

    elapsed = time.time() - start
    logger.info(f"DALL-E 3: Done in {elapsed:.1f}s, {len(img_resp.content) / 1024:.1f} KB")
    return img_resp.content
