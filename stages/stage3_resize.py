from pathlib import Path
from PIL import Image
from utils.logger import get_logger
from utils.config import load_config

logger = get_logger(__name__)

PRINT_SIZES = {
    "5x7":   (1500, 2100),
    "8x10":  (2400, 3000),
    "11x14": (3300, 4200),
    "16x20": (4800, 6000),
    "18x24": (5400, 7200),
    "24x36": (7200, 10800),
    "A4":    (2480, 3508),
    "A3":    (3508, 4961),
}


def _centre_crop_and_resize(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Crop to the target aspect ratio from the centre, then scale to exact dimensions."""
    src_w, src_h = img.size
    target_ratio = target_w / target_h
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        # Source is wider — crop left and right equally
        new_w = int(src_h * target_ratio)
        offset = (src_w - new_w) // 2
        img = img.crop((offset, 0, offset + new_w, src_h))
    elif src_ratio < target_ratio:
        # Source is taller — crop top and bottom equally
        new_h = int(src_w / target_ratio)
        offset = (src_h - new_h) // 2
        img = img.crop((0, offset, src_w, offset + new_h))

    return img.resize((target_w, target_h), Image.LANCZOS)


def resize_to_all_sizes(
    source_path: str | Path,
    output_dir: str | Path,
    sizes: list[str] | None = None,
) -> list[str]:
    """
    Resize source image to all requested print dimensions.
    Returns list of generated size names.
    """
    config = load_config()
    if sizes is None:
        sizes = config.get("generate_sizes", list(PRINT_SIZES.keys()))

    jpeg_quality = config.get("jpeg_quality", 97)
    max_upscale = config.get("max_upscale_factor", 3.0)

    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []

    with Image.open(source_path) as src:
        src_w, src_h = src.size
        src_rgb = src.convert("RGB")
        logger.info(f"Stage 3: Source image {src_w}x{src_h}, generating {len(sizes)} sizes")

        for size_name in sizes:
            if size_name not in PRINT_SIZES:
                logger.warning(f"Stage 3: Unknown size '{size_name}', skipping")
                continue

            target_w, target_h = PRINT_SIZES[size_name]

            # Warn if upscaling more than max_upscale_factor
            upscale_factor = max(target_w / src_w, target_h / src_h)
            if upscale_factor > max_upscale:
                logger.warning(
                    f"Stage 3: {size_name} requires {upscale_factor:.1f}x upscale "
                    f"(max recommended: {max_upscale}x) — generating anyway"
                )

            resized = _centre_crop_and_resize(src_rgb, target_w, target_h)

            filename = f"print_{size_name}_{target_w}x{target_h}px.jpg"
            out_path = output_dir / filename
            resized.save(
                out_path,
                format="JPEG",
                quality=jpeg_quality,
                dpi=(300, 300),
            )

            file_kb = out_path.stat().st_size / 1024
            logger.info(f"Stage 3: {filename} — {target_w}x{target_h}px, {file_kb:.0f} KB")

            if file_kb > 50 * 1024:
                logger.warning(
                    f"Stage 3: {filename} is {file_kb / 1024:.1f} MB — "
                    f"above GitHub 50 MB recommendation"
                )

            generated.append(size_name)

    logger.info(f"Stage 3: Complete — {len(generated)} sizes generated")
    return generated
