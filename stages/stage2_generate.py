import time
from pathlib import Path
from utils.config import load_config, get_env
from utils.logger import get_logger
from utils.image_validation import is_valid_image_file, validate_dimensions

logger = get_logger(__name__)

DRY_RUN_IMAGE_SIZE = 512
MOCK_IMAGE_PATH = "tests/mock_image.jpg"

PROVIDER_COSTS = {
    "fal": 0.050,
    "dalle3": 0.120,
    "ideogram": 0.080,
}


def _get_mock_image() -> bytes:
    """Return a minimal valid JPEG for dry-run mode."""
    mock_path = Path(MOCK_IMAGE_PATH)
    if mock_path.exists():
        return mock_path.read_bytes()

    from PIL import Image
    import io
    img = Image.new("RGB", (768, 1152), color=(180, 160, 140))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def generate_image(
    enhanced_prompt: str,
    output_path: str | Path,
    dry_run: bool = False,
) -> tuple[str, float]:
    """
    Returns (provider_name, estimated_cost_usd).
    Writes the downloaded image to output_path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.info("[DRY RUN] Using mock image instead of calling image API")
        image_bytes = _get_mock_image()
        output_path.write_bytes(image_bytes)
        return "dry_run", 0.0

    config = load_config()
    max_retries = config.get("max_retries_per_stage", 2)
    backoff = config.get("retry_backoff_seconds", 15)
    preferred = config.get("image_provider", "fal")

    provider_order = _build_provider_order(preferred)

    for provider_name in provider_order:
        for attempt in range(max_retries + 1):
            try:
                image_bytes = _call_provider(provider_name, enhanced_prompt)

                # Write to temp path for validation
                output_path.write_bytes(image_bytes)

                valid, reason = is_valid_image_file(output_path)
                if not valid:
                    logger.warning(f"Stage 2: Image validation failed ({reason}) — retrying")
                    if attempt < max_retries:
                        time.sleep(backoff)
                        continue
                    break  # Try next provider

                dim_valid, dim_info = validate_dimensions(output_path)
                if not dim_valid:
                    logger.warning(f"Stage 2: Dimension check failed ({dim_info}) — retrying")
                    if attempt < max_retries:
                        time.sleep(backoff)
                        continue
                    break

                cost = PROVIDER_COSTS.get(provider_name, 0.0)
                logger.info(f"Stage 2: Success with {provider_name}, {dim_info}, cost ~${cost:.3f}")
                return provider_name, cost

            except Exception as e:
                logger.warning(f"Stage 2: {provider_name} attempt {attempt + 1} failed: {e}")
                if attempt < max_retries:
                    time.sleep(backoff)
                else:
                    logger.error(f"Stage 2: {provider_name} exhausted retries, trying next provider")
                    break

    raise RuntimeError("Stage 2: All image providers failed — skipping today's generation")


def _build_provider_order(preferred: str) -> list[str]:
    """Only include providers whose API keys are configured."""
    import os
    key_map = {
        "fal": "FAL_API_KEY",
        "dalle3": "OPENAI_API_KEY",
        "ideogram": "IDEOGRAM_API_KEY",
    }
    all_providers = ["fal", "dalle3", "ideogram"]
    available = [p for p in all_providers if os.getenv(key_map[p], "").strip()]

    if not available:
        raise RuntimeError("No image provider API keys are configured in .env")

    if preferred in available:
        available.remove(preferred)
        return [preferred] + available
    return available


def _call_provider(name: str, prompt: str) -> bytes:
    if name == "fal":
        from providers.fal_ai import generate
    elif name == "dalle3":
        from providers.dalle3 import generate
    elif name == "ideogram":
        from providers.ideogram import generate
    else:
        raise ValueError(f"Unknown provider: {name}")
    return generate(prompt)
