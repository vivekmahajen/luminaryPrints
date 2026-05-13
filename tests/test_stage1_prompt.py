import pytest
from stages.stage1_prompt import validate_prompt, enhance_prompt

VALID_PROMPT = (
    "A breathtaking textured abstract painting in oil on canvas, featuring thick impasto "
    "brushstrokes and bold palette knife marks in warm champagne, ivory and soft gold tones. "
    "Soft diffused natural light illuminates the richly textured surface, following the rule "
    "of thirds with a strong centred focal point. Visible canvas texture beneath the paint "
    "layers evokes the Contemporary abstract expressionism tradition. Portrait 2:3 format "
    "composition with dynamic diagonal energy and three-dimensional depth. "
    "Photorealistic painting detail, ultra high resolution, 300 DPI print quality, "
    "museum quality fine art, Etsy wall art bestseller."
)


def test_valid_prompt_passes():
    ok, errors = validate_prompt(VALID_PROMPT)
    assert ok, f"Valid prompt failed: {errors}"


def test_prompt_too_short():
    short = "A painting with oil on canvas, 300 DPI print quality, museum quality fine art."
    ok, errors = validate_prompt(short)
    assert not ok
    assert any("short" in e.lower() or "80" in e for e in errors)


def test_forbidden_word_watermark():
    bad = VALID_PROMPT.replace("palette knife", "watermark logo")
    ok, errors = validate_prompt(bad)
    assert not ok
    assert any("watermark" in e for e in errors)


def test_missing_quality_markers():
    no_quality = (
        "A painting in oil on canvas with thick impasto brushstrokes and palette knife marks. "
        "Soft diffused light, rule of thirds composition, visible canvas texture. "
        "Contemporary abstract expressionism, portrait 2:3 format, ultra high resolution only."
    )
    ok, errors = validate_prompt(no_quality)
    assert not ok


def test_dry_run_returns_valid_prompt():
    prompt = enhance_prompt(
        style_name="Textured Abstract",
        style_base_description="Thick impasto oil painting",
        colour_variation="Warm champagne and ivory",
        technique_hint="palette knife",
        dry_run=True,
    )
    assert len(prompt.split()) >= 30
    ok, errors = validate_prompt(prompt)
    assert ok, f"Dry-run prompt failed validation: {errors}"
