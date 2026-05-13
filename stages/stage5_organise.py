import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from utils.logger import get_logger
from utils.config import load_config

logger = get_logger(__name__)


def build_folder_name(date_str: str, style_id: str) -> str:
    return f"{date_str}_{style_id}"


def organise_outputs(
    date_str: str,
    style_id: str,
    style_name: str,
    colour_variation: str,
    enhanced_prompt: str,
    base_prompt: str,
    source_image_path: str | Path,
    sizes_generated: list[str],
    etsy_metadata: dict,
    image_provider: str,
    generation_time_seconds: float,
    total_pipeline_time_seconds: float,
    estimated_cost_usd: float,
    image_url_original: str = "",
    status: str = "success",
) -> Path:
    config = load_config()
    output_base = Path(config.get("output_dir", "output"))

    folder_name = build_folder_name(date_str, style_id)
    folder_path = output_base / folder_name

    # Avoid collisions
    if folder_path.exists():
        timestamp = int(time.time())
        folder_path = output_base / f"{folder_name}_{timestamp}"
        logger.warning(f"Stage 5: Output folder existed — using {folder_path.name}")

    folder_path.mkdir(parents=True, exist_ok=True)

    source_image_path = Path(source_image_path)
    if config.get("keep_source_image", True) and source_image_path.exists():
        dest_source = folder_path / "source_original.jpg"
        shutil.copy2(source_image_path, dest_source)
        logger.info(f"Stage 5: Copied source image to {dest_source}")

    # Move all print_ files from the same directory as source to output folder
    source_dir = source_image_path.parent
    for print_file in source_dir.glob("print_*.jpg"):
        dest = folder_path / print_file.name
        shutil.move(str(print_file), str(dest))
    logger.info(f"Stage 5: Moved print files to {folder_path}")

    # Write etsy_listing.json
    etsy_path = folder_path / "etsy_listing.json"
    etsy_path.write_text(json.dumps(etsy_metadata, indent=2, ensure_ascii=False))
    logger.info("Stage 5: Written etsy_listing.json")

    # Write social_media.json
    social = {
        "instagram_caption": etsy_metadata.get("instagram_caption", ""),
        "pinterest_caption": etsy_metadata.get("pinterest_caption", ""),
    }
    (folder_path / "social_media.json").write_text(json.dumps(social, indent=2))

    # Write generation_log.json
    log_data = {
        "date": date_str,
        "style": style_name,
        "variation": colour_variation,
        "prompt_raw": base_prompt,
        "prompt_enhanced": enhanced_prompt,
        "image_provider": image_provider,
        "image_url_original": image_url_original,
        "source_image_size_kb": round(
            (folder_path / "source_original.jpg").stat().st_size / 1024, 1
        ) if (folder_path / "source_original.jpg").exists() else 0,
        "source_image_dimensions": _get_dimensions(folder_path / "source_original.jpg"),
        "sizes_generated": sizes_generated,
        "generation_time_seconds": round(generation_time_seconds, 2),
        "total_pipeline_time_seconds": round(total_pipeline_time_seconds, 2),
        "estimated_cost_usd": round(estimated_cost_usd, 4),
        "status": status,
    }
    (folder_path / "generation_log.json").write_text(json.dumps(log_data, indent=2))
    logger.info(f"Stage 5: Written generation_log.json — status={status}")

    return folder_path


def _get_dimensions(path: Path) -> str:
    if not path.exists():
        return "unknown"
    try:
        from PIL import Image
        with Image.open(path) as img:
            return f"{img.width}x{img.height}"
    except Exception:
        return "unknown"


def update_index(
    output_dir: str | Path,
    date_str: str,
    style_name: str,
    colour_variation: str,
    folder_name: str,
):
    output_dir = Path(output_dir)
    index_path = output_dir / "index.md"

    new_row = f"| {date_str} | {style_name} | {colour_variation} | [View](./{folder_name}/) |"

    if index_path.exists():
        content = index_path.read_text()
        lines = content.splitlines()
        # Insert new row after the header rows (first 3 lines)
        insert_pos = 3
        for i, line in enumerate(lines):
            if line.startswith("|") and i >= 3:
                insert_pos = i
                break
        lines.insert(insert_pos, new_row)
        index_path.write_text("\n".join(lines) + "\n")
    else:
        header = (
            "# Daily Art Prints — All Generated Prints\n\n"
            "| Date | Style | Variation | Folder |\n"
            "|------|-------|-----------|--------|\n"
            f"{new_row}\n"
        )
        index_path.write_text(header)

    logger.info("Stage 5: Updated index.md")
