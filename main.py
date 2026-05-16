#!/usr/bin/env python3
"""
ArtPilot Daily — AI Art Print Generator
Entry point. Orchestrates all 6 pipeline stages.

Usage:
  python main.py                                    # Normal run (next in rotation)
  python main.py --dry-run                          # No API calls, no GitHub push
  python main.py --style textured_abstract          # Override style
  python main.py --variation "warm champagne"       # Override variation
  python main.py --schedule                         # Run on schedule (cron-replacement)

  # Custom portrait modes:
  python main.py --custom-image pet.jpg --portrait-type pet
  python main.py --custom-image photo.jpg --portrait-type faceless
  python main.py --custom-image pet.jpg --portrait-type pet --dry-run

  # Generate room mockups for the latest (or a specific) output folder:
  python main.py --mockup
  python main.py --mockup --mockup-folder output/2026-05-13_custom_pet_portrait_1778727476
"""

import argparse
import json
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from utils.config import load_config, load_styles
from utils.database import init_db, insert_run, update_run, mark_variation_used
from utils.logger import get_logger
from utils.alerts import send_failure_alert

from stages.stage1_prompt import enhance_prompt
from stages.stage2_generate import generate_image
from stages.stage3_resize import resize_to_all_sizes
from stages.stage4_metadata import generate_metadata
from stages.stage5_organise import organise_outputs, update_index
from stages.stage6_github import push_to_github
from stages.stage_custom_portrait import (
    build_custom_prompt, get_next_variation,
    advance_variation_state, PORTRAIT_TYPES, PORTRAIT_STRENGTH,
)
from stages.stage_mockup import generate_mockups

logger = get_logger("main")

STATE_PATH = Path("state.json")


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"current_style_index": 0, "current_variation_index": 0, "total_runs": 0, "last_run_date": None}


def save_state(state: dict):
    STATE_PATH.write_text(json.dumps(state, indent=2))


def advance_state(state: dict, num_styles: int, num_variations: int) -> dict:
    vi = state["current_variation_index"] + 1
    si = state["current_style_index"]
    if vi >= num_variations:
        vi = 0
        si = (si + 1) % num_styles
    state["current_style_index"] = si
    state["current_variation_index"] = vi
    state["total_runs"] = state.get("total_runs", 0) + 1
    state["last_run_date"] = datetime.now().date().isoformat()
    return state


def pick_style_and_variation(
    styles: list[dict],
    state: dict,
    config: dict,
    style_override: str | None = None,
    variation_override: str | None = None,
) -> tuple[dict, str]:
    rotation = config.get("style_rotation_order", [s["id"] for s in styles])

    if style_override:
        style = next((s for s in styles if s["id"] == style_override), None)
        if not style:
            raise ValueError(f"Unknown style '{style_override}'. Valid: {[s['id'] for s in styles]}")
    else:
        style_id = rotation[state["current_style_index"] % len(rotation)]
        style = next((s for s in styles if s["id"] == style_id), styles[0])

    if variation_override:
        variation = variation_override
    else:
        vi = state["current_variation_index"] % len(style["variations"])
        variation = style["variations"][vi]

    return style, variation


def run_pipeline(
    dry_run: bool = False,
    style_override: str | None = None,
    variation_override: str | None = None,
) -> bool:
    pipeline_start = time.time()
    config = load_config()
    styles = load_styles()
    init_db()
    state = load_state()

    style, variation = pick_style_and_variation(
        styles, state, config, style_override, variation_override
    )

    date_str = datetime.now().date().isoformat()
    logger.info(f"=== ArtPilot Daily | {date_str} | {style['name']} | {variation} ===")

    run_id = insert_run({
        "date": date_str,
        "style_id": style["id"],
        "style_name": style["name"],
        "variation": variation,
    })

    output_dir = Path(config.get("output_dir", "output"))
    output_dir.mkdir(exist_ok=True)

    try:
        # ── Stage 1: Prompt Enhancement ──────────────────────────────────────
        logger.info("── Stage 1: Prompt enhancement")
        t1 = time.time()
        base_prompt = f"{style['name']} — {variation}"
        enhanced_prompt = enhance_prompt(
            style_name=style["name"],
            style_base_description=style["base_description"],
            colour_variation=variation,
            technique_hint=style.get("technique_hint", ""),
            dry_run=dry_run,
        )
        logger.info(f"Stage 1 done in {time.time() - t1:.1f}s")
        update_run(run_id, {"prompt_enhanced": enhanced_prompt})

        # ── Stage 2: Image Generation ─────────────────────────────────────────
        logger.info("── Stage 2: Image generation")
        t2 = time.time()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_source = Path(tmpdir) / "source_original.jpg"

            provider, cost = generate_image(
                enhanced_prompt=enhanced_prompt,
                output_path=tmp_source,
                dry_run=dry_run,
            )
            generation_time = time.time() - t2
            logger.info(f"Stage 2 done in {generation_time:.1f}s — provider={provider}")
            update_run(run_id, {"image_provider": provider, "source_image_path": str(tmp_source)})

            # ── Stage 3: Resize ───────────────────────────────────────────────
            logger.info("── Stage 3: Resizing to print dimensions")
            t3 = time.time()
            sizes_generated = resize_to_all_sizes(
                source_path=tmp_source,
                output_dir=Path(tmpdir),
            )
            logger.info(f"Stage 3 done in {time.time() - t3:.1f}s — {len(sizes_generated)} sizes")

            # ── Stage 4: Etsy Metadata ────────────────────────────────────────
            logger.info("── Stage 4: Etsy metadata generation")
            t4 = time.time()
            metadata = generate_metadata(
                style_name=style["name"],
                colour_variation=variation,
                enhanced_prompt=enhanced_prompt,
                sizes_generated=sizes_generated,
                dry_run=dry_run,
            )
            logger.info(f"Stage 4 done in {time.time() - t4:.1f}s")
            update_run(run_id, {"etsy_title": metadata.get("title", "")})

            # ── Stage 5: Organise ─────────────────────────────────────────────
            logger.info("── Stage 5: Organising output files")
            total_time = time.time() - pipeline_start
            folder_path = organise_outputs(
                date_str=date_str,
                style_id=style["id"],
                style_name=style["name"],
                colour_variation=variation,
                enhanced_prompt=enhanced_prompt,
                base_prompt=base_prompt,
                source_image_path=tmp_source,
                sizes_generated=sizes_generated,
                etsy_metadata=metadata,
                image_provider=provider,
                generation_time_seconds=generation_time,
                total_pipeline_time_seconds=total_time,
                estimated_cost_usd=cost,
                status="success",
            )
            logger.info(f"Stage 5 done — folder: {folder_path}")

        update_index(output_dir, date_str, style["name"], variation, folder_path.name)
        update_run(run_id, {"output_folder": str(folder_path), "sizes_generated": ",".join(sizes_generated)})

        # ── Stage 6: GitHub Push ──────────────────────────────────────────────
        logger.info("── Stage 6: Pushing to GitHub")
        t6 = time.time()
        github_url = push_to_github(
            folder_path=folder_path,
            date_str=date_str,
            style_name=style["name"],
            dry_run=dry_run,
        )
        logger.info(f"Stage 6 done in {time.time() - t6:.1f}s — {github_url}")

        total_time = time.time() - pipeline_start
        update_run(run_id, {
            "github_url": github_url,
            "generation_time_sec": generation_time,
            "estimated_cost_usd": cost,
            "status": "success",
        })
        mark_variation_used(style["id"], variation, date_str)

        # Advance rotation state (only if not using overrides)
        if not style_override and not variation_override:
            new_state = advance_state(state, len(styles), len(style["variations"]))
            save_state(new_state)

        logger.info(f"=== Pipeline complete in {total_time:.1f}s | Cost ~${cost:.3f} ===")
        return True

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        update_run(run_id, {"status": "failed", "error_message": str(e)})
        if config.get("alert_on_failure", True):
            send_failure_alert(
                subject=f"Pipeline failed — {date_str}",
                body=f"Style: {style['name']}\nVariation: {variation}\nError: {e}",
            )
        return False


def run_custom_portrait(
    image_path: str,
    portrait_type: str,
    dry_run: bool = False,
) -> bool:
    pipeline_start = time.time()
    config = load_config()
    init_db()
    state = load_state()

    pt_info = PORTRAIT_TYPES.get(portrait_type)
    if not pt_info:
        logger.error(f"Unknown portrait type '{portrait_type}'. Use: pet or faceless")
        return False

    variation = get_next_variation(portrait_type, state)
    style_name = pt_info["label"]
    style_id = pt_info["style_id"]
    date_str = datetime.now().date().isoformat()

    logger.info(f"=== Custom Portrait | {date_str} | {style_name} | {variation} ===")
    logger.info(f"Reference image: {image_path}")

    run_id = insert_run({
        "date": date_str,
        "style_id": style_id,
        "style_name": style_name,
        "variation": variation,
    })

    output_dir = Path(config.get("output_dir", "output"))
    output_dir.mkdir(exist_ok=True)

    try:
        # Stage 1: Generate portrait-specific prompt
        logger.info("── Stage 1: Custom portrait prompt")
        t1 = time.time()
        enhanced_prompt = build_custom_prompt(portrait_type, variation, dry_run=dry_run)
        logger.info(f"Stage 1 done in {time.time() - t1:.1f}s")
        update_run(run_id, {"prompt_enhanced": enhanced_prompt})

        # Stage 2: Image-to-image generation using reference photo
        logger.info("── Stage 2: Image-to-image generation")
        t2 = time.time()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_source = Path(tmpdir) / "source_original.jpg"

            if dry_run:
                from stages.stage2_generate import generate_image
                provider, cost = generate_image(enhanced_prompt, tmp_source, dry_run=True)
            else:
                from providers.fal_ai_img2img import upload_image_to_fal, generate_custom_portrait
                ref_url = upload_image_to_fal(image_path)
                image_bytes = generate_custom_portrait(
                    enhanced_prompt, ref_url,
                    reference_local_path=str(image_path),
                    strength=PORTRAIT_STRENGTH,
                )
                tmp_source.write_bytes(image_bytes)
                provider, cost = "fal_redux", 0.05

            generation_time = time.time() - t2
            logger.info(f"Stage 2 done in {generation_time:.1f}s")
            update_run(run_id, {"image_provider": provider})

            # Stage 3: Resize
            logger.info("── Stage 3: Resizing to print dimensions")
            sizes_generated = resize_to_all_sizes(tmp_source, Path(tmpdir))

            # Stage 4: Metadata
            logger.info("── Stage 4: Etsy metadata")
            metadata = generate_metadata(
                style_name=style_name,
                colour_variation=variation,
                enhanced_prompt=enhanced_prompt,
                sizes_generated=sizes_generated,
                dry_run=dry_run,
            )
            update_run(run_id, {"etsy_title": metadata.get("title", "")})

            # Stage 5: Organise
            logger.info("── Stage 5: Organising files")
            folder_path = organise_outputs(
                date_str=date_str,
                style_id=style_id,
                style_name=style_name,
                colour_variation=variation,
                enhanced_prompt=enhanced_prompt,
                base_prompt=f"{style_name} — {variation}",
                source_image_path=tmp_source,
                sizes_generated=sizes_generated,
                etsy_metadata=metadata,
                image_provider=provider,
                generation_time_seconds=generation_time,
                total_pipeline_time_seconds=time.time() - pipeline_start,
                estimated_cost_usd=cost,
                status="success",
            )

        update_index(output_dir, date_str, style_name, variation, folder_path.name)
        update_run(run_id, {"output_folder": str(folder_path), "sizes_generated": ",".join(sizes_generated)})

        # Stage 6: GitHub push
        logger.info("── Stage 6: Pushing to GitHub")
        github_url = push_to_github(folder_path, date_str, style_name, dry_run=dry_run)

        update_run(run_id, {"github_url": github_url, "status": "success",
                            "generation_time_sec": generation_time, "estimated_cost_usd": cost})

        # Stage 7: Room mockups
        if not dry_run:
            logger.info("── Stage 7: Generating room mockups")
            try:
                mockup_paths = generate_mockups(folder_path)
                logger.info(f"Stage 7 done — {len(mockup_paths)} mockup(s) written")
            except Exception as e:
                logger.warning(f"Stage 7 mockup generation failed (non-fatal): {e}")

        # Advance variation state
        new_state = advance_variation_state(portrait_type, state)
        save_state(new_state)

        logger.info(f"=== Custom portrait complete in {time.time() - pipeline_start:.1f}s ===")
        logger.info(f"Output folder: {folder_path}")
        return True

    except Exception as e:
        logger.exception(f"Custom portrait pipeline failed: {e}")
        update_run(run_id, {"status": "failed", "error_message": str(e)})
        if config.get("alert_on_failure", True):
            send_failure_alert(
                subject=f"Custom portrait failed — {date_str}",
                body=f"Type: {portrait_type}\nImage: {image_path}\nError: {e}",
            )
        return False


def _latest_output_folder(config: dict) -> Path:
    output_dir = Path(config.get("output_dir", "output"))
    folders = sorted(
        (p for p in output_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not folders:
        raise FileNotFoundError(f"No output folders found in {output_dir}")
    return folders[0]


def main():
    parser = argparse.ArgumentParser(description="ArtPilot Daily — AI Art Print Generator")
    parser.add_argument("--dry-run", action="store_true", help="Run all stages without API calls or GitHub push")
    parser.add_argument("--style", type=str, help="Override style (e.g. textured_abstract)")
    parser.add_argument("--variation", type=str, help="Override colour variation")
    parser.add_argument("--schedule", action="store_true", help="Run on schedule defined in config.json")
    parser.add_argument("--custom-image", type=str, help="Path to reference photo for custom portrait")
    parser.add_argument("--portrait-type", type=str, choices=["pet", "faceless"],
                        help="Type of custom portrait: pet or faceless")
    parser.add_argument("--mockup", action="store_true",
                        help="Generate room mockup images for the latest output folder")
    parser.add_argument("--mockup-folder", type=str,
                        help="Path to a specific output folder to generate mockups for")
    args = parser.parse_args()

    # Standalone mockup mode
    if args.mockup or args.mockup_folder:
        config = load_config()
        if args.mockup_folder:
            folder = Path(args.mockup_folder)
        else:
            folder = _latest_output_folder(config)
        logger.info(f"=== Mockup mode | folder: {folder.name} ===")
        try:
            written = generate_mockups(folder)
            for p in written:
                logger.info(f"  → {p.name}")
            logger.info(f"=== Mockup complete — {len(written)} image(s) ===")
            sys.exit(0)
        except Exception as e:
            logger.exception(f"Mockup failed: {e}")
            sys.exit(1)

    # Custom portrait mode
    if args.custom_image:
        if not args.portrait_type:
            print("Error: --portrait-type is required with --custom-image (use: pet or faceless)")
            sys.exit(1)
        image_path = Path(args.custom_image)
        if not args.dry_run and not image_path.exists():
            print(f"Error: Image file not found: {image_path}")
            sys.exit(1)
        success = run_custom_portrait(
            image_path=str(image_path),
            portrait_type=args.portrait_type,
            dry_run=args.dry_run,
        )
        sys.exit(0 if success else 1)

    if args.schedule:
        import schedule as sched

        config = load_config()
        run_time = config.get("schedule_time", "07:00")
        logger.info(f"Scheduling daily run at {run_time}")

        sched.every().day.at(run_time).do(run_pipeline)
        while True:
            sched.run_pending()
            time.sleep(30)
    else:
        success = run_pipeline(
            dry_run=args.dry_run,
            style_override=args.style,
            variation_override=args.variation,
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
