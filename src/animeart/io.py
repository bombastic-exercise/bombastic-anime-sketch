"""Small artifact utilities; PNG exports need only NumPy and the standard library."""

import base64
import html
import json
import struct
import zlib
from pathlib import Path

import numpy as np


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_png(path, pixels):
    pixels = np.asarray(pixels)
    if pixels.dtype != np.uint8:
        pixels = np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8)
    if pixels.ndim == 2:
        color = 0
    elif pixels.ndim == 3 and pixels.shape[2] == 3:
        color = 2
    else:
        raise ValueError("PNG must be H×W grayscale or H×W×3 RGB")
    height, width = pixels.shape[:2]

    def chunk(kind, data):
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))

    payload = b"".join(b"\0" + row.tobytes() for row in pixels)
    data = b"\x89PNG\r\n\x1a\n"
    data += chunk(b"IHDR", struct.pack("!2I5B", width, height, 8, color, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress(payload)) + chunk(b"IEND", b"")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def gallery(path, rows):
    """Write a portable HTML comparison sheet with embedded images and captions."""
    cards = []
    for title, images in rows:
        figures = []
        for label, image_path in images:
            data = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
            figures.append(f'<figure><img src="data:image/png;base64,{data}"><figcaption>{html.escape(label)}</figcaption></figure>')
        cards.append(f"<section><h2>{html.escape(title)}</h2><div>{''.join(figures)}</div></section>")
    Path(path).write_text("<!doctype html><meta charset='utf-8'><title>AnimeArt experiments</title>"
                          "<style>body{background:#ededeb;color:#222;font:15px system-ui;margin:32px}"
                          "section>div{display:flex;flex-wrap:wrap;gap:16px}figure{margin:0}"
                          "img{width:256px;image-rendering:auto;border:1px solid #bbb}"
                          "figcaption{max-width:256px;padding:8px 0}h2{font-size:17px}</style>"
                          "<h1>AnimeArt · rendering experiments</h1>" + "".join(cards))
