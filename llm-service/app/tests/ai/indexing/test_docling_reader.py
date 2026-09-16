import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
from llama_index.core.node_parser import SentenceSplitter

from app.ai.indexing.base import BaseTextIndexer
from app.ai.indexing.readers import docling_reader as dr
from app.ai.indexing.readers.base_reader import ChunksResult
from app.ai.indexing.readers.docling_reader import DoclingReader
from app.ai.indexing.readers.pdf import PDFReader
from app.ai.indexing.readers.pptx import PptxReader


class DummyIndexer(BaseTextIndexer):
    def index_file(self, file_path: Path, doc_id: str) -> None:  # pragma: no cover
        raise NotImplementedError


def make_reader() -> DoclingReader:
    return DoclingReader(
        splitter=SentenceSplitter(chunk_size=512, chunk_overlap=0),
        document_id="doc-1",
        data_source_id=1,
    )


# --- Reader selection (_get_reader_class) -------------------------------


def test_get_reader_class_routes_slides_to_docling_when_enhanced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("USE_ENHANCED_PDF_PROCESSING", "true")
    indexer = DummyIndexer(1)

    for filename in ("doc.pdf", "deck.pptx", "deck.pptm", "page.html"):
        assert (
            indexer._get_reader_class(Path(filename)) is DoclingReader
        ), f"expected {filename} to route to DoclingReader"

    # Non-docling extensions must still use their default readers.
    assert indexer._get_reader_class(Path("notes.docx")) is not DoclingReader


def test_get_reader_class_uses_default_readers_when_not_enhanced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("USE_ENHANCED_PDF_PROCESSING", "false")
    indexer = DummyIndexer(1)

    assert indexer._get_reader_class(Path("doc.pdf")) is PDFReader
    assert indexer._get_reader_class(Path("deck.pptx")) is PptxReader
    assert indexer._get_reader_class(Path("deck.pptm")) is PptxReader


# --- _convert_pptx_to_pdf -----------------------------------------------


def test_convert_pptx_to_pdf_requires_libreoffice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dr, "_find_soffice", lambda: None)
    pptx = tmp_path / "slides.pptx"
    pptx.write_bytes(b"fake")

    with pytest.raises(RuntimeError, match="LibreOffice"):
        DoclingReader._convert_pptx_to_pdf(pptx)


def test_convert_pptx_to_pdf_invokes_soffice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        dr,
        "_find_soffice",
        lambda: "/home/cdsw/libreoffice_local/opt/libreoffice7.6/program/soffice.bin",
    )
    # Pin the profile dir so the function doesn't probe /tmp or touch real $HOME.
    monkeypatch.setattr(dr, "_writable_profile_dir", lambda: tmp_path)

    pptx = tmp_path / "slides.pptx"
    pptx.write_bytes(b"fake")

    invoked: list[list[str]] = []
    invoked_env: list[dict[str, str]] = []

    def fake_run(
        cmd: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess:
        invoked.append(cmd)
        invoked_env.append(kwargs.get("env", {}))  # type: ignore[arg-type]
        outdir = cmd[cmd.index("--outdir") + 1]
        (Path(outdir) / "slides.pdf").write_bytes(b"pdf")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    pdf_path, temp_dir = DoclingReader._convert_pptx_to_pdf(pptx)
    try:
        assert pdf_path.exists()
        assert pdf_path.name == "slides.pdf"

        cmd = invoked[0]
        # Must use soffice.bin directly (bypasses oosplash/X11 launcher).
        assert cmd[0] == (
            "/home/cdsw/libreoffice_local/opt/libreoffice7.6/program/soffice.bin"
        )
        assert cmd[1].startswith("-env:UserInstallation=file://")
        assert "--headless" in cmd
        assert "--norestore" in cmd
        assert "--invisible" in cmd
        assert "--convert-to" in cmd
        assert "pdf" in cmd
        assert cmd[cmd.index("--outdir") + 1] == str(temp_dir)

        # Headless VCL plugin must be forced so no X11 libs are loaded.
        assert invoked_env and invoked_env[0].get("SAL_USE_VCLPLUGIN") == "headless"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# --- load_chunks ordering (PPTX -> PDF -> Docling OCR) ------------------


def _patch_docling_components(
    monkeypatch: pytest.MonkeyPatch, converted_paths: list[Path]
) -> None:
    class FakeDoclingDocument:
        def iterate_items(self):
            return iter([])

    class FakeDocumentConverter:
        def __init__(self, **kwargs: object) -> None:
            pass

        def convert(self, path: Path):
            converted_paths.append(Path(path))
            return SimpleNamespace(document=FakeDoclingDocument())

    class FakeHybridChunker:
        def __init__(self, **kwargs: object) -> None:
            pass

        def chunk(self, doc: object):
            item = SimpleNamespace(prov=[SimpleNamespace(page_no=2)])
            chunk = SimpleNamespace(
                text="Hello world slide text",
                meta=SimpleNamespace(doc_items=[item]),
            )
            return [chunk]

    monkeypatch.setattr(dr, "DocumentConverter", FakeDocumentConverter)
    monkeypatch.setattr(
        dr.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **k: object())
    )
    monkeypatch.setattr(dr, "HybridChunker", FakeHybridChunker)


def test_load_chunks_converts_pptx_before_docling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    converted_paths: list[Path] = []
    _patch_docling_components(monkeypatch, converted_paths)

    rendered_pdf = Path("/tmp/fake_rendered.pdf")
    temp_dir = Path("/tmp/fake_tmp")
    monkeypatch.setattr(
        DoclingReader,
        "_convert_pptx_to_pdf",
        staticmethod(lambda fp: (rendered_pdf, temp_dir)),
    )

    pptx = tmp_path / "deck.pptx"
    pptx.write_bytes(b"fake")

    result = make_reader().load_chunks(pptx)

    # Docling OCR must run on the rendered PDF, not the original pptx.
    assert converted_paths == [rendered_pdf]
    assert len(result.chunks) == 1
    assert result.chunks[0].metadata["file_name"] == "deck.pptx"
    assert result.chunks[0].metadata["page_number"] == 2


def test_load_chunks_passes_pdf_through_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    converted_paths: list[Path] = []
    _patch_docling_components(monkeypatch, converted_paths)

    convert_called: list[bool] = []

    def fake_convert(fp: Path):
        convert_called.append(True)
        raise AssertionError("_convert_pptx_to_pdf should not be called for PDFs")

    monkeypatch.setattr(DoclingReader, "_convert_pptx_to_pdf", staticmethod(fake_convert))

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"fake")

    result = make_reader().load_chunks(pdf)

    assert convert_called == []
    assert converted_paths == [pdf]
    assert len(result.chunks) == 1


# --- Qwen OCR engine (load_chunks with ENHANCED_PDF_ENGINE=qwen) ----------


def test_load_chunks_qwen_uses_ocr_for_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENHANCED_PDF_ENGINE", "qwen")
    monkeypatch.setattr(
        dr,
        "ocr_pdf",
        lambda pdf_path: [(1, "First page text."), (2, "Second page text.")],
    )

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"fake")

    result = make_reader().load_chunks(pdf)

    assert result.chunks
    assert sorted(c.metadata["page_number"] for c in result.chunks) == [1, 2]
    assert all(c.metadata["file_name"] == "doc.pdf" for c in result.chunks)
    assert all(c.metadata["document_id"] == "doc-1" for c in result.chunks)


def test_load_chunks_qwen_converts_pptx_before_ocr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENHANCED_PDF_ENGINE", "qwen")

    rendered_pdf = Path("/tmp/fake_rendered_for_qwen.pdf")
    temp_dir = Path("/tmp/fake_tmp_qwen")
    monkeypatch.setattr(
        DoclingReader,
        "_convert_pptx_to_pdf",
        staticmethod(lambda fp: (rendered_pdf, temp_dir)),
    )

    ocr_calls: list[Path] = []

    def fake_ocr(pdf_path: Path) -> list[tuple[int, str]]:
        ocr_calls.append(Path(pdf_path))
        return [(1, "slide text")]

    monkeypatch.setattr(dr, "ocr_pdf", fake_ocr)

    pptx = tmp_path / "deck.pptx"
    pptx.write_bytes(b"fake")

    result = make_reader().load_chunks(pptx)

    assert ocr_calls == [rendered_pdf]
    assert result.chunks
    assert result.chunks[0].metadata["file_name"] == "deck.pptx"
    assert result.chunks[0].metadata["page_number"] == 1


def test_load_chunks_qwen_skips_empty_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENHANCED_PDF_ENGINE", "qwen")
    monkeypatch.setattr(
        dr,
        "ocr_pdf",
        lambda pdf_path: [(1, "   "), (2, "Real content")],
    )

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"fake")

    result = make_reader().load_chunks(pdf)

    assert result.chunks
    assert all(c.metadata["page_number"] == 2 for c in result.chunks)


def test_load_chunks_qwen_not_used_for_html(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENHANCED_PDF_ENGINE", "qwen")
    ocr_called: list[Path] = []
    monkeypatch.setattr(
        dr,
        "ocr_pdf",
        lambda pdf_path: (ocr_called.append(Path(pdf_path)) or [(1, "never")]),
    )
    docling_called: list[Path] = []

    def fake_docling(fp: Path) -> ChunksResult:
        docling_called.append(fp)
        return ChunksResult([])

    monkeypatch.setattr(DoclingReader, "_load_chunks_docling", fake_docling)

    html = tmp_path / "page.html"
    html.write_bytes(b"<html><body>hello</body></html>")

    result = make_reader().load_chunks(html)

    # HTML always routes to the docling engine; Qwen OCR must never run.
    assert ocr_called == []
    assert docling_called == [html]
    assert result.chunks == []