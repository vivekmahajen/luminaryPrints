import pytest
from pathlib import Path
import tempfile
from stages.stage2_generate import generate_image


def test_dry_run_creates_valid_jpeg():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "source_original.jpg"
        provider, cost = generate_image(
            enhanced_prompt="A test painting prompt",
            output_path=out_path,
            dry_run=True,
        )
        assert provider == "dry_run"
        assert cost == 0.0
        assert out_path.exists()
        assert out_path.stat().st_size > 0

        # Verify JPEG magic bytes
        with open(out_path, "rb") as f:
            magic = f.read(3)
        assert magic == b"\xff\xd8\xff", "Output is not a valid JPEG"


def test_dry_run_provider_order_unchanged():
    # Dry run should not interact with providers at all
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "img.jpg"
        provider, _ = generate_image("prompt", out, dry_run=True)
        assert provider == "dry_run"
