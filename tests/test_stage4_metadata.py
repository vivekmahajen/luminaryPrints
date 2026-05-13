import pytest
from stages.stage4_metadata import generate_metadata, _validate_metadata


def test_dry_run_returns_valid_structure():
    metadata = generate_metadata(
        style_name="Textured Abstract",
        colour_variation="Warm champagne and ivory",
        enhanced_prompt="An impasto oil painting...",
        sizes_generated=["5x7", "8x10", "11x14", "16x20", "18x24", "24x36", "A4", "A3"],
        dry_run=True,
    )
    assert isinstance(metadata, dict)
    assert "title" in metadata
    assert "description" in metadata
    assert "tags" in metadata
    assert len(metadata["tags"]) == 13


def test_metadata_validation_passes_on_valid_data():
    metadata = generate_metadata(
        style_name="Floral Pop",
        colour_variation="Giant cobalt blue peonies",
        enhanced_prompt="Bold oversized floral painting...",
        sizes_generated=["8x10", "16x20"],
        dry_run=True,
    )
    ok, errors = _validate_metadata(metadata)
    assert ok, f"Dry-run metadata failed validation: {errors}"


def test_validation_catches_short_title():
    bad = {"title": "Too short", "description": "x" * 400, "tags": ["a"] * 13}
    ok, errors = _validate_metadata(bad)
    assert not ok
    assert any("Title" in e for e in errors)


def test_validation_catches_wrong_tag_count():
    long_desc = (
        "Transform your home with this stunning wall art. Beautiful colours create atmosphere. "
        "✨ WHAT YOU GET: 8 JPEG files. Sizes: 8x10, 16x20. Resolution: 300 DPI. "
        "🖨️ HOW TO PRINT: home printer. 🎁 PERFECT FOR: living room. "
        "⚡ INSTANT DOWNLOAD. No physical item is shipped."
    )
    bad = {
        "title": "Printable Wall Art Abstract Print | Digital Download Instant Download | Modern Home Decor Art",
        "description": long_desc,
        "tags": ["a", "b"],  # wrong count
    }
    ok, errors = _validate_metadata(bad)
    assert not ok
    assert any("13" in e for e in errors)


def test_all_required_description_elements():
    metadata = generate_metadata(
        style_name="Nature and Botanical",
        colour_variation="Fern frond study",
        enhanced_prompt="Botanical watercolour...",
        sizes_generated=["8x10"],
        dry_run=True,
    )
    desc = metadata["description"].lower()
    assert "300 dpi" in desc
    assert "instant download" in desc or "digital download" in desc
    assert "no physical item" in desc or "no physical" in desc
