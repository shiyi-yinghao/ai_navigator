from __future__ import annotations
import base64
import mimetypes
from pathlib import Path
from typing import Any

import httpx

from ai_navigator.infra.exceptions import PreProcessorError
from ai_navigator.infra.models import ContentPart

SUPPORTED_MIME_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/gif", "image/webp"}
)


class ImageProcessor:
    """Prepare images for multimodal LLM requests.

    Normalises local files, remote URLs, and raw bytes into ContentPart objects
    that can be included in a Message's ``content`` list.
    """

    def from_path(self, path: str | Path) -> ContentPart:
        """Load and base64-encode a local image file."""
        p = Path(path)
        if not p.exists():
            raise PreProcessorError(f"Image file not found: {path}")
        mime, _ = mimetypes.guess_type(str(p))
        if mime not in SUPPORTED_MIME_TYPES:
            raise PreProcessorError(
                f"Unsupported image type '{mime}'. "
                f"Supported: {sorted(SUPPORTED_MIME_TYPES)}"
            )
        data = base64.b64encode(p.read_bytes()).decode()
        return ContentPart(type="image_base64", image_data=data, media_type=mime)

    def from_url(self, url: str) -> ContentPart:
        """Reference an image by public URL without downloading it."""
        return ContentPart(type="image_url", image_url=url)

    def from_url_download(self, url: str, timeout: float = 30.0) -> ContentPart:
        """Download an image from a URL and encode it as base64."""
        try:
            resp = httpx.get(url, follow_redirects=True, timeout=timeout)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise PreProcessorError(
                f"Failed to download image from {url}: {exc}"
            ) from exc
        content_type = (
            resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
        )
        data = base64.b64encode(resp.content).decode()
        return ContentPart(type="image_base64", image_data=data, media_type=content_type)

    def from_bytes(self, data: bytes, media_type: str = "image/jpeg") -> ContentPart:
        """Encode raw image bytes as base64."""
        return ContentPart(
            type="image_base64",
            image_data=base64.b64encode(data).decode(),
            media_type=media_type,
        )

    def resize(
        self,
        path: str | Path,
        max_px: int = 1024,
        output_format: str | None = None,
    ) -> ContentPart:
        """Resize an image so that neither dimension exceeds ``max_px``, then encode.

        Requires ``pillow``: pip install ai-navigator[image]
        """
        try:
            from PIL import Image
        except ImportError as exc:
            raise PreProcessorError(
                "pillow is required for resize. "
                "Install with: pip install ai-navigator[image]"
            ) from exc
        import io

        img = Image.open(str(path))
        img.thumbnail((max_px, max_px))
        fmt = output_format or img.format or "JPEG"
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        mime = f"image/{fmt.lower().replace('jpeg', 'jpeg')}"
        if mime == "image/jpg":
            mime = "image/jpeg"
        return self.from_bytes(buf.getvalue(), media_type=mime)
