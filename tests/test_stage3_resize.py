import pytest
from pathlib import Path
import tempfile
from PIL import Image
from stages.stage3_resize import resize_to_all_sizes, PRINT_SIZES, _centre_crop_and_resize


def _make_test_image(path: Path, w: int = 1024, h: int = 1536):
    img = Image.new("RGB", (w, h), color=(100, 150, 200))
    img.save(path, format="JPEG", quality=85)


def test_resize_produces_correct_dimensions():
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "source.jpg"
        _make_test_image(src)
        out_dir = Path(tmpdir) / "output"

        generated = resize_to_all_sizes(src, out_dir, sizes=["5x7", "8x10"])

        assert "5x7" in generated
        assert "8x10" in generated

        img_5x7 = Image.open(out_dir / "print_5x7_1500x2100px.jpg")
        assert img_5x7.size == (1500, 2100)

        img_8x10 = Image.open(out_dir / "print_8x10_2400x3000px.jpg")
        assert img_8x10.size == (2400, 3000)


def test_centre_crop_wider_source():
    img = Image.new("RGB", (2000, 1000))
    result = _centre_crop_and_resize(img, 500, 750)
    assert result.size == (500, 750)


def test_centre_crop_taller_source():
    img = Image.new("RGB", (1000, 3000))
    result = _centre_crop_and_resize(img, 1500, 2100)
    assert result.size == (1500, 2100)


def test_all_print_sizes_have_correct_filenames():
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "source.jpg"
        _make_test_image(src)
        out_dir = Path(tmpdir) / "out"

        generated = resize_to_all_sizes(src, out_dir, sizes=list(PRINT_SIZES.keys()))
        assert len(generated) == len(PRINT_SIZES)

        for size_name, (w, h) in PRINT_SIZES.items():
            expected = out_dir / f"print_{size_name}_{w}x{h}px.jpg"
            assert expected.exists(), f"Missing: {expected.name}"


def test_dpi_metadata_set():
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "source.jpg"
        _make_test_image(src)
        out_dir = Path(tmpdir) / "out"
        resize_to_all_sizes(src, out_dir, sizes=["8x10"])

        img = Image.open(out_dir / "print_8x10_2400x3000px.jpg")
        dpi = img.info.get("dpi")
        assert dpi == (300, 300), f"DPI metadata not set correctly: {dpi}"
