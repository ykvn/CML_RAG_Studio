#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2025
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. ("Cloudera") to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#

import base64
import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import httpx

from ....config import settings

logger = logging.getLogger(__name__)

QWEN_OCR_MODEL = "Qwen3.8-27B-ocr"
OCR_DPI = 150
OCR_TIMEOUT = 600.0

# One page is sent per request, so no PAGE markers are needed. Crucially this
# prompt forbids summarizing/truncating tables or lists, which previously caused
# table rows to be dropped when the model ran out of output tokens.
OCR_PAGE_PROMPT = (
    "Perform OCR of the single page of the document shown in the image. "
    "Return all the text exactly as it appears, preserving layout where possible. "
    "IMPORTANT: Return every table row and every list item verbatim; do not "
    "summarize, truncate, or omit any table rows. "
    "If the page contains any charts or graphs, provide a concise summary of them "
    "separately at the end, clearly labeled 'CHART SUMMARY:'. "
    "Do not summarize or omit any table data."
)

_PAGE_MARKER_RE = re.compile(r"---\s*PAGE\s+(\d+)\s*---", re.IGNORECASE)


def pdf_to_base64_images(pdf_path: Path, dpi: int = OCR_DPI) -> List[str]:
    """Render every PDF page to a base64-encoded JPEG."""
    import fitz  # PyMuPDF - imported lazily so module import never requires it

    encoded_images: List[str] = []
    doc = fitz.open(str(pdf_path))
    try:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            image_bytes = pix.tobytes("jpeg")
            encoded_images.append(base64.b64encode(image_bytes).decode("utf-8"))
    finally:
        doc.close()
    return encoded_images


def _split_pages(content: str) -> List[Tuple[int, str]]:
    """Split OCR content into (page_number, text) using '--- PAGE X ---' markers."""
    matches = list(_PAGE_MARKER_RE.finditer(content))
    if not matches:
        if content.strip():
            return [(1, content.strip())]
        return []

    pages: List[Tuple[int, str]] = []
    for i, m in enumerate(matches):
        page_number = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        text = content[start:end].strip()
        if text:
            pages.append((page_number, text))
    return pages


def _stream_ocr(
    client: httpx.Client,
    url: str,
    headers: dict,
    payload: dict,
    label: str,
) -> Tuple[str, Optional[str]]:
    """Stream a single ChatCompletion request.

    Returns (content_text, finish_reason). Only the model's ``content`` (the
    actual OCR text) is retained; ``reasoning_content`` is intentionally
    discarded. Emits a WARNING when the response is truncated because the
    model hit its output-token limit (finish_reason == "length").
    """
    content_text = ""
    finish_reason: Optional[str] = None
    with client.stream("POST", url, headers=headers, json=payload) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                content_text += content
            finish_reason = choice.get("finish_reason") or finish_reason

    if finish_reason == "length":
        logger.warning(
            "Qwen OCR %s was truncated (finish_reason='length'): only %d chars "
            "received. Some content may be missing.",
            label,
            len(content_text),
        )

    return content_text, finish_reason


def ocr_pdf(
    pdf_path: Path,
    model: str = QWEN_OCR_MODEL,
    dpi: int = OCR_DPI,
) -> List[Tuple[int, str]]:
    """Run Qwen OCR over a PDF, one request per page.

    Each page is sent as its own request so that every page/table gets the model's
    full output-token budget, avoiding mid-table truncation. Returns a list of
    (page_number, text) for every non-empty page.
    ``reasoning_content`` is discarded; ``finish_reason`` is logged.
    """
    base_url = (settings.openai_api_base or "").rstrip("/")
    api_key = settings.openai_api_key
    if not base_url or not api_key:
        raise RuntimeError(
            "Qwen OCR requires OPENAI_API_BASE and OPENAI_API_KEY to be configured."
        )

    images_b64 = pdf_to_base64_images(pdf_path, dpi=dpi)
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}

    logger.info("Qwen OCR: processing %d page(s)", len(images_b64))

    results: List[Tuple[int, str]] = []
    with httpx.Client(
        verify=False, timeout=httpx.Timeout(OCR_TIMEOUT, connect=10.0)
    ) as client:
        for page_number, img in enumerate(images_b64, start=1):
            content_payload: List[dict] = [
                {"type": "text", "text": OCR_PAGE_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                },
            ]
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": content_payload}],
                "stream": True,
                "context_window": 32768,
            }
            label = f"page {page_number}/{len(images_b64)}"
            content_text, _ = _stream_ocr(client, url, headers, payload, label)
            logger.info(
                "Qwen OCR %s done: %d char(s), finish_reason captured",
                label,
                len(content_text),
            )
            if content_text.strip():
                results.append((page_number, content_text.strip()))

    if not results and images_b64:
        logger.warning("Qwen OCR produced no text for %d page(s).", len(images_b64))

    return results
