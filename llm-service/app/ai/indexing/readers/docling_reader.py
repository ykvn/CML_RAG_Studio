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

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, PictureItem
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.base import BaseChunk
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from llama_index.core.schema import Document, NodeRelationship, TextNode
from transformers import AutoTokenizer

from .base_reader import BaseReader, ChunksResult
from .pdf import MarkdownSerializerProvider

logger = logging.getLogger(__name__)

# Offline Model Paths inside CDSW Environment
DOCLING_ARTIFACTS_PATH = Path("/home/cdsw/llm-service/models/docling_models")
TOKENIZER_PATH = "/home/cdsw/llm-service/models/bge-m3-tokenizer"


def clean_ocr_kerning(text: str) -> str:
    """Safely repairs OCR character splitting without altering regular text."""
    # 1. Replace non-printable private Unicode replacement characters (\ue353 -> :)
    cleaned = re.sub(r"[\ue000-\uf8ff]", ":", text)

    # 2. Safely join sequences of 2+ isolated uppercase letters (e.g., "M A W X" -> "MAWX", "A B C" -> "ABC")
    cleaned = re.sub(r"\b[A-Z](?:\s+[A-Z])+\b", lambda m: m.group(0).replace(" ", ""), cleaned)

    return cleaned


def _candidate_exists(path: Optional[Path]) -> bool:
    """Return True if ``path`` is an existing, executable file."""
    return bool(path) and path.is_file() and os.access(path, os.X_OK)


def _writable_profile_dir() -> Path:
    """Create a writable LibreOffice profile location.

    Prefers the OS default temp dir, but many hardened CML/CDSW runtimes mount
    ``/tmp`` with ``noexec``, which makes LibreOffice fail at startup (exit code
    81, empty stderr) because it cannot execute helpers it extracts into the
    profile. In that case we fall back to a writable subdir under the user's
    home so headless conversion works.
    """
    try:
        tmp = Path(tempfile.mkdtemp(prefix="lo_profile_"))
        probe = tmp / "probe.sh"
        probe.write_text("#!/bin/sh\nexit 0\n")
        probe.chmod(0o700)
        subprocess.run(
            [str(probe)], capture_output=True, timeout=10, check=False
        )
        shutil.rmtree(tmp, ignore_errors=True)
        return tmp.parent  # /tmp is executable
    except Exception:
        # /tmp is not executable-usable -> use a home-directory subdir.
        home = Path(os.environ.get("HOME", str(Path.home())))
        profile_base = home / ".libreoffice_rag_profile"
        profile_base.mkdir(parents=True, exist_ok=True)
        return profile_base


def _find_soffice() -> Optional[str]:
    """Locate a LibreOffice binary, preferring the real ``soffice.bin`` engine.

    The main ``soffice``/``soffice`` wrapper launches ``oosplash``, which loads
    X11 libraries (e.g. ``libXinerama``) unavailable in headless CDSW sessions,
    so we strongly prefer ``soffice.bin`` when present.

    Search order:
      1. Explicit CDSW locations (independent of PATH/launcher environment).
      2. The process ``PATH`` (``soffice.bin`` then ``soffice`` then ``libreoffice``).
    """
    # Well-known extracted-LibreOffice installs under the CDSW user home.
    libreoffice_local = Path("/home/cdsw/libreoffice_local")
    known_dirs = [libreoffice_local / "opt"]
    known_dirs.extend(sorted(p for p in libreoffice_local.glob("*/opt")))
    # Include a flat layout and any `.../program` dirs for robustness.
    globs: list[Path] = []
    for base in known_dirs:
        globs.extend(base.glob("*/program/soffice.bin"))
        globs.extend(base.glob("*/*/program/soffice.bin"))

    for candidate in sorted(set(globs), key=str):
        if _candidate_exists(candidate):
            return str(candidate)

    # Standard PATH search, again preferring the real binary, then wrappers.
    for name in ("soffice.bin", "soffice", "libreoffice"):
        found = shutil.which(name)
        if found:
            return found

    return None


class DoclingReader(BaseReader):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @staticmethod
    def _convert_pptx_to_pdf(file_path: Path) -> tuple[Path, Path]:
        soffice = _find_soffice()
        if not soffice:
            raise RuntimeError(
                "LibreOffice ('soffice') is required to OCR PowerPoint (.pptx/.pptm) "
                "files. Install LibreOffice and ensure 'soffice.bin'/'soffice' is on "
                "PATH (e.g. `apt-get install libreoffice-impress`)."
            )

        temp_dir = Path(tempfile.mkdtemp(prefix="docling_pptx_"))
        
        # Use a profile location on an executable filesystem
        profile_dir = _writable_profile_dir() / f"lo_profile_{os.getpid()}"
        profile_dir.mkdir(parents=True, exist_ok=True)
        
        env = os.environ.copy()
        env["SAL_USE_VCLPLUGIN"] = "headless"
        env["HOME"] = str(temp_dir)
        env["USERPROFILE"] = str(temp_dir)
        env["LO_USERPROFILE"] = str(profile_dir)

        # Inject the libreoffice program directory into LD_LIBRARY_PATH 
        soffice_dir = str(Path(soffice).parent)
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{soffice_dir}:{existing_ld}".strip(":")

        cmd = [
            soffice,
            "-env:UserInstallation=file://"
            + str(profile_dir).replace(" ", "%20"),
            "--headless",
            "--norestore",
            "--nofirststartwizard",
            "--invisible",
            "--convert-to",
            "pdf",
            "--outdir",
            str(temp_dir),
            str(file_path),
        ]

        try:
            # FIX: explicitly pass env=env
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
            )
            
            # Catch exit code 81 (profile creation) and retry once automatically
            if result.returncode == 81:
                logger.info("LibreOffice returned code 81 (profile created). Retrying conversion...")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env=env,
                )
                
        except subprocess.TimeoutExpired as exc:
            shutil.rmtree(profile_dir, ignore_errors=True)
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(
                f"LibreOffice timed out converting '{file_path.name}' to PDF."
            ) from exc

        if result.returncode != 0:
            detail = (result.stderr or "").strip() or (result.stdout or "").strip()
            shutil.rmtree(profile_dir, ignore_errors=True)
            shutil.rmtree(temp_dir, ignore_errors=True)
            if not detail:
                detail = (
                    "LibreOffice exited during startup/initialisation (empty "
                    "output). This often indicates the profile location is on a "
                    "noexec filesystem or a stale soffice process holds a lock. "
                    f"Command: {result.args!r}"
                )
            raise RuntimeError(
                f"LibreOffice failed to convert '{file_path.name}' to PDF for "
                f"enhanced processing: {detail}"
            )

        pdf_path = temp_dir / f"{file_path.stem}.pdf"
        if not pdf_path.exists():
            shutil.rmtree(profile_dir, ignore_errors=True)
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(
                f"LibreOffice did not produce a PDF output for '{file_path.name}'."
            )
            
        shutil.rmtree(profile_dir, ignore_errors=True)
        return pdf_path, temp_dir

    def load_chunks(self, file_path: Path) -> ChunksResult:
        document = Document()
        document.id_ = self.document_id
        self._add_document_metadata(document, file_path)
        parent = document.as_related_node_info()

        # PPTX/PPTM slides need to be rendered to PDF (via LibreOffice) so that the
        # Docling OCR pipeline can see text embedded in slide images. Native PDFs are
        # processed directly.
        is_slides = file_path.suffix.lower() in {".pptx", ".pptm"}
        doc_path = file_path
        temp_dir: Optional[Path] = None
        if is_slides:
            doc_path, temp_dir = self._convert_pptx_to_pdf(file_path)

        try:
            # 1. Offline Pipeline Options: Point to local model artifacts & upscale image scale
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            pipeline_options.images_scale = 3.0  # High-definition 3x scale to prevent OCR blurring
            pipeline_options.artifacts_path = DOCLING_ARTIFACTS_PATH  # Offline model directory
            pipeline_options.ocr_options = EasyOcrOptions()

            converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF],
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                },
            )

            logger.debug(f"Processing {doc_path.suffix} with Docling: {doc_path=}")
            docling_doc: ConversionResult = converter.convert(doc_path)
        finally:
            #if temp_dir is not None:
            #    shutil.rmtree(temp_dir, ignore_errors=True)
            pass

        # 2. Chart Extraction: Recover numerical text locked inside Picture / Bar Chart items
        for item, _ in docling_doc.document.iterate_items():
            if isinstance(item, PictureItem):
                if hasattr(item, "annotations") and item.annotations:
                    chart_text = " ".join(
                        [ann.text for ann in item.annotations if hasattr(ann, "text")]
                    )
                    if chart_text:
                        item.text = f"[CHART DATA]: {chart_text}"

        # 3. Hybrid Chunker: Load local bge-m3 tokenizer & configure 4096 token limit
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

        chunker = HybridChunker(
            serializer_provider=MarkdownSerializerProvider(),
            tokenizer=tokenizer,
            max_tokens=2048,
            merge_peers=True,
        )
        chunky_chunks = chunker.chunk(docling_doc.document)

        converted_chunks: List[TextNode] = []

        for i, chunky_chunk in enumerate(chunky_chunks):
            page_number: int = 0
            if not hasattr(chunky_chunk.meta, "doc_items"):
                continue

            raw_text = chunky_chunk.text

            # 4. Filter Empty Borders: Skip chunks containing only hyphens and pipes
            clean_check = (
                raw_text.replace("|", "").replace("-", "").replace("\n", "").strip()
            )
            if not clean_check:
                logger.warning(
                    f"Chunk {i} contains only empty markdown formatting, skipping."
                )
                continue

            # 5. Kerning Normalization: Clean up split spaces in OCR output
            normalized_text = clean_ocr_kerning(raw_text)

            for item in chunky_chunk.meta.doc_items:
                page_number = item.prov[0].page_no if item.prov else None

            node = TextNode(text=normalized_text)
            if page_number:
                node.metadata["page_number"] = page_number
            node.metadata["file_name"] = document.metadata["file_name"]
            node.metadata["document_id"] = document.metadata["document_id"]
            node.metadata["data_source_id"] = document.metadata["data_source_id"]
            node.metadata["chunk_number"] = i
            node.metadata["chunk_format"] = "markdown"
            node.relationships.update({NodeRelationship.SOURCE: parent})

            converted_chunks.append(node)

        return ChunksResult(converted_chunks)