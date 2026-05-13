import struct
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG"
MIN_FILE_SIZE_KB = 200
MIN_WIDTH = 512
MIN_HEIGHT = 768


def is_valid_image_file(path: str | Path) -> tuple[bool, str]:
    path = Path(path)

    if not path.exists():
        return False, "File does not exist"

    size_kb = path.stat().st_size / 1024
    if size_kb < MIN_FILE_SIZE_KB:
        return False, f"File too small: {size_kb:.1f} KB (min {MIN_FILE_SIZE_KB} KB)"

    with open(path, "rb") as f:
        header = f.read(8)

    if not (header[:3] == JPEG_MAGIC or header[:4] == PNG_MAGIC):
        return False, f"Not a valid JPEG or PNG (magic bytes: {header[:4].hex()})"

    return True, "ok"


def validate_dimensions(path: str | Path, min_width: int = MIN_WIDTH, min_height: int = MIN_HEIGHT) -> tuple[bool, str]:
    try:
        from PIL import Image
        with Image.open(path) as img:
            w, h = img.size
            if w < min_width or h < min_height:
                return False, f"Dimensions too small: {w}x{h} (min {min_width}x{min_height})"
            return True, f"{w}x{h}"
    except Exception as e:
        return False, str(e)
