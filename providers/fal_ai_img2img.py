import io
import os
import time
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
from utils.config import get_env
from utils.logger import get_logger

logger = get_logger(__name__)

FAL_IMG2IMG_MODEL = "fal-ai/flux/dev/image-to-image"


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


def _create_premasked_input(local_path: str | Path) -> bytes:
    """
    Create a version of the reference image where the background is replaced
    with neutral grey. This guides img2img to paint over the grey areas as
    abstract art while the detailed subject in the center is preserved.

    Subject oval (centre 72% x 78%): original photo pixels — kept sharp
    Background:                       neutral grey — model paints abstract art
    """
    with Image.open(local_path) as img:
        img = img.convert("RGB")
        w, h = img.size

    # Soft oval mask: white=subject (keep), black=background (replace)
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    subject_w = int(w * 0.72)
    subject_h = int(h * 0.78)
    x0 = (w - subject_w) // 2
    y0 = (h - subject_h) // 2
    draw.ellipse([x0, y0, x0 + subject_w, y0 + subject_h], fill=255)
    feather = max(int(min(w, h) * 0.12), 20)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))

    grey_bg = Image.new("RGB", (w, h), (135, 135, 135))
    with Image.open(local_path) as img:
        subject = img.convert("RGB")
    result = Image.composite(subject, grey_bg, mask)

    buf = io.BytesIO()
    result.save(buf, format="JPEG", quality=92)
    logger.info(f"Pre-masked input: {w}x{h}, subject oval {subject_w}x{subject_h}")
    return buf.getvalue()


def generate_custom_portrait(
    prompt: str,
    reference_image_url: str,
    reference_local_path: str = "",
    strength: float = 0.85,
) -> bytes:
    """
    Pre-masked img2img approach:

    1. Replace background of reference photo with neutral grey
       (subject oval stays as original photo pixels)
    2. Submit to fal-ai/flux/dev/image-to-image at strength=0.65
    3. Model transforms grey background → abstract art from prompt
       while the detailed subject guides preservation of the center

    This avoids the fal-ai/flux-pro/v1.1/fill endpoint which completes
    in <0.1s with no output (broken inpainting endpoint).
    """
    api_key = get_env("FAL_API_KEY")
    os.environ["FAL_KEY"] = api_key
    import fal_client

    local_path = reference_local_path if reference_local_path else None

    if not local_path or not Path(local_path).exists():
        logger.info("Downloading reference image")
        img_bytes = requests.get(reference_image_url, timeout=60).content
        local_path = "_tmp_portrait_ref.jpg"
        Path(local_path).write_bytes(img_bytes)

    logger.info("Creating pre-masked input (subject on grey background)")
    premasked_bytes = _create_premasked_input(local_path)
    premasked_tmp = "_tmp_portrait_premasked.jpg"
    Path(premasked_tmp).write_bytes(premasked_bytes)
    premasked_url = fal_client.upload_file(premasked_tmp)
    Path(premasked_tmp).unlink(missing_ok=True)
    logger.info(f"Pre-masked image uploaded — {premasked_url}")

    abstract_prompt = (
        f"{prompt} The areas surrounding the central subject are an expressive abstract oil painting "
        f"with bold impasto brushstrokes, swirling palette knife marks, and rich saturated colour. "
        f"The central subject remains photorealistic and sharp against the painterly background."
    )

    arguments = {
        "image_url": premasked_url,
        "prompt": abstract_prompt,
        "strength": 0.65,
        "num_inference_steps": 28,
        "guidance_scale": 7.5,
        "num_images": 1,
        "enable_safety_checker": True,
        "output_format": "jpeg",
    }

    auth_headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}

    logger.info("fal.ai img2img: Submitting job")
    start = time.time()
    submit_resp = requests.post(
        f"https://queue.fal.run/{FAL_IMG2IMG_MODEL}",
        json=arguments,
        headers=auth_headers,
        timeout=30,
    )
    submit_resp.raise_for_status()
    job = submit_resp.json()
    request_id = job["request_id"]
    status_url = job.get("status_url") or f"https://queue.fal.run/fal-ai/flux/requests/{request_id}/status"
    response_url = job.get("response_url") or f"https://queue.fal.run/fal-ai/flux/requests/{request_id}"
    logger.info(f"fal.ai img2img: request_id={request_id}, status_url={status_url}")

    # Poll status
    for poll in range(80):
        time.sleep(3)
        st = requests.get(status_url, headers=auth_headers, timeout=15)
        st.raise_for_status()
        body = st.json()
        status = body.get("status", "")
        inference_time = body.get("metrics", {}).get("inference_time", 0)
        logger.info(f"fal.ai img2img: Poll {poll + 1} — {status} (inference={inference_time:.2f}s)")
        if status == "COMPLETED":
            logger.info(f"fal.ai img2img: Completed! inference={inference_time:.2f}s")
            break
        if status in ("FAILED", "ERROR"):
            raise RuntimeError(f"fal.ai img2img: Job failed — {body}")
    else:
        raise TimeoutError("fal.ai img2img: Timed out after 240s")

    # Fetch result — try fal.ai's provided URL first, then versioned path
    result = None
    for url in [response_url, f"https://queue.fal.run/{FAL_IMG2IMG_MODEL}/requests/{request_id}"]:
        r = requests.get(url, headers=auth_headers, timeout=30)
        logger.info(f"fal.ai img2img: GET {url} → {r.status_code} Allow={r.headers.get('Allow', '-')}")
        if r.status_code == 200:
            result = r.json()
            break
    if result is None:
        raise RuntimeError("fal.ai img2img: Could not retrieve result from any URL")

    # Handle various response shapes
    images = result.get("images") or result.get("output", {}).get("images") or []
    if images:
        image_url = images[0].get("url") if isinstance(images[0], dict) else images[0]
    elif result.get("image"):
        image_url = result["image"].get("url") if isinstance(result["image"], dict) else result["image"]
    elif result.get("url"):
        image_url = result["url"]
    else:
        raise RuntimeError(f"fal.ai img2img: No image URL in response: {result}")

    logger.info(f"fal.ai img2img: Downloading result from {image_url}")
    img = requests.get(image_url, timeout=60)
    img.raise_for_status()

    elapsed = time.time() - start
    logger.info(f"fal.ai img2img: Done in {elapsed:.1f}s, {len(img.content) // 1024} KB")

    if reference_local_path == "" and Path("_tmp_portrait_ref.jpg").exists():
        Path("_tmp_portrait_ref.jpg").unlink(missing_ok=True)

    return img.content
