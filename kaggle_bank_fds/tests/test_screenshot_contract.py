"""Structural contracts for reproducible portfolio screenshots."""

from __future__ import annotations

from pathlib import Path
import struct
import zlib


ROOT = Path(__file__).parents[2]
IMAGE_DIR = ROOT / "docs" / "images"
SCREENSHOTS = (
    "bank-fds-upload-preview.png",
    "bank-fds-risk-result.png",
    "bank-fds-alert-history.png",
)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _decode_png(path: Path) -> tuple[int, int, bytes]:
    content = path.read_bytes()
    assert content.startswith(PNG_SIGNATURE)
    offset = len(PNG_SIGNATURE)
    width = height = 0
    compressed = bytearray()
    saw_end = False
    while offset < len(content):
        length = struct.unpack(">I", content[offset:offset + 4])[0]
        kind = content[offset + 4:offset + 8]
        data = content[offset + 8:offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height = struct.unpack(">II", data[:8])
        elif kind == b"IDAT":
            compressed.extend(data)
        elif kind == b"IEND":
            saw_end = True
            break
    decoded = zlib.decompress(bytes(compressed))
    assert saw_end and offset == len(content)
    return width, height, decoded


def _chunk_types(path: Path) -> tuple[bytes, ...]:
    content = path.read_bytes()
    offset = len(PNG_SIGNATURE)
    chunks = []
    while offset < len(content):
        length = struct.unpack(">I", content[offset:offset + 4])[0]
        chunks.append(content[offset + 4:offset + 8])
        offset += 12 + length
    return tuple(chunks)


def test_three_named_png_screenshots_are_present_and_decodable():
    assert tuple(sorted(path.name for path in IMAGE_DIR.glob("*.png"))) == tuple(
        sorted(SCREENSHOTS)
    )
    for filename in SCREENSHOTS:
        width, height, decoded = _decode_png(IMAGE_DIR / filename)
        assert width >= 1_000 and height >= 700
        assert decoded and len(set(decoded[:100_000])) > 1


def test_screenshot_dimensions_and_sizes_are_portfolio_reasonable():
    sizes = []
    dimensions = []
    for filename in SCREENSHOTS:
        path = IMAGE_DIR / filename
        width, height, _ = _decode_png(path)
        sizes.append(path.stat().st_size)
        dimensions.append((width, height))
        assert path.stat().st_size < 2 * 1024 * 1024
        assert width <= 2_000 and height <= 2_000
    assert sum(sizes) < 5 * 1024 * 1024
    assert len(set(dimensions)) <= len(dimensions)


def test_screenshots_contain_no_optional_metadata_chunks():
    for filename in SCREENSHOTS:
        chunks = _chunk_types(IMAGE_DIR / filename)
        assert set(chunks) == {b"IHDR", b"IDAT", b"IEND"}


def test_documentation_references_real_screenshots():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "demo-guide.md").read_text(encoding="utf-8")
    assert "![Exact overlap 분석 결과](docs/images/bank-fds-risk-result.png)" in readme
    for filename in SCREENSHOTS:
        assert f"](images/{filename})" in guide


def test_no_extra_screenshots_database_or_macos_metadata_in_project_tree():
    image_files = tuple(
        path for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    assert set(image_files) == {IMAGE_DIR / filename for filename in SCREENSHOTS}
    assert not tuple(ROOT.rglob(".DS_Store"))
    assert not tuple(
        path for path in ROOT.rglob("*")
        if path.is_file() and (
            path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}
            or path.name.endswith((".sqlite3-wal", ".sqlite3-shm"))
        )
    )
