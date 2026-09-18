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
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from ....config import settings

logger = logging.getLogger(__name__)

QWEN_OCR_MODEL = "Qwen3.8-27B-ocr"
OCR_DPI = 150
OCR_TIMEOUT = 600.0
MAX_OCR_PAGE_CHARS = 12000  # client-side backstop: terminates stream on hallucinated org-chart
                             # loops (e.g. repeating names) and keeps garbage low (~3k tokens).
                             # 12k chars ≈ 2.4× the largest legitimate page seen in logs, so
                             # real dense table/list pages are preserved.

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
    "Do not format the output as HTML or XML; STRICTLY return plain text."
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
            if len(content_text) >= MAX_OCR_PAGE_CHARS:
                logger.warning(
                    "Qwen OCR %s hit the %d-char output cap (possible repetition loop "
                    "or pathological output); terminating the stream to avoid an "
                    "indefinite hang. finish_reason=%s, received=%d chars.",
                    label, MAX_OCR_PAGE_CHARS,
                    finish_reason or "none", len(content_text),
                )
                break

    if finish_reason == "length":
        logger.warning(
            "Qwen OCR %s was truncated (finish_reason='length'): only %d chars "
            "received. Some content may be missing.",
            label,
            len(content_text),
        )

    return content_text, finish_reason


def _process_single_page(
    client: httpx.Client,
    page_number: int,
    img_b64: str,
    total_pages: int,
    model: str,
    url: str,
    headers: dict,
) -> Tuple[int, str]:
    """Worker function to process a single page via the API."""
    content_payload = [
        {"type": "text", "text": OCR_PAGE_PROMPT},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        },
    ]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content_payload}],
        "stream": True,
        #"context_window": 32768,
    }
    label = f"page {page_number}/{total_pages}"
    
    try:
        content_text, _ = _stream_ocr(client, url, headers, payload, label)
        logger.info(
            "Qwen OCR %s done: %d char(s), finish_reason captured",
            label,
            len(content_text),
        )
        return page_number, content_text.strip()
    except Exception as e:
        logger.error("Qwen OCR %s failed: %s", label, e)
        return page_number, ""


def ocr_pdf(
    pdf_path: Path,
    model: str = QWEN_OCR_MODEL,
    dpi: int = OCR_DPI,
) -> List[Tuple[int, str]]:
    """Run Qwen OCR over a PDF concurrently.

    Pages are sent in parallel via a ThreadPoolExecutor to drastically reduce 
    overall processing time, and the results are sorted sequentially afterward.
    """
    base_url = (settings.openai_api_base or "").rstrip("/")
    api_key = settings.openai_api_key
    if not base_url or not api_key:
        raise RuntimeError(
            "Qwen OCR requires OPENAI_API_BASE and OPENAI_API_KEY to be configured."
        )

    images_b64 = pdf_to_base64_images(pdf_path, dpi=dpi)
    total_pages = len(images_b64)
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}

    logger.info("Qwen OCR: processing %d page(s) concurrently", total_pages)

    results_dict = {}
    
    # Process up to 10 pages concurrently to maximize H200 throughput without overwhelming gateway timeouts
    max_workers = min(2, total_pages) if total_pages > 0 else 1

    with httpx.Client(
        verify=False, timeout=httpx.Timeout(OCR_TIMEOUT, connect=10.0)
    ) as client:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _process_single_page, 
                    client, 
                    page_number, 
                    img, 
                    total_pages, 
                    model, 
                    url, 
                    headers
                ): page_number
                for page_number, img in enumerate(images_b64, start=1)
            }

            for future in as_completed(futures):
                page_num, text_content = future.result()
                if text_content:
                    results_dict[page_num] = text_content

    if not results_dict and images_b64:
        logger.warning("Qwen OCR produced no text for %d page(s).", total_pages)

    # Sort results by page number to guarantee chunks remain in sequential order
    sorted_results = [(pn, results_dict[pn]) for pn in sorted(results_dict.keys())]
    return sorted_results