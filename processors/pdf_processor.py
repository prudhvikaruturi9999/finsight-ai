from pathlib import Path

from processors.base_processor import BaseProcessor
from enrichment.metadata_schema import ParsedDocument, ParsedPage
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_MAX_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024


class PdfProcessor(BaseProcessor):
    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".pdf"

    def parse(self, file_path: str) -> ParsedDocument:
        path = Path(file_path)
        size_bytes = path.stat().st_size

        if size_bytes > _MAX_BYTES:
            logger.warning(
                f"Skipping {path.name}: {size_bytes / 1e6:.1f} MB exceeds "
                f"{settings.MAX_FILE_SIZE_MB} MB limit"
            )
            return ParsedDocument(source_path=str(path), doc_type="pdf", title=path.stem)

        logger.info(f"Parsing {path.name} ({size_bytes / 1e6:.1f} MB)")

        # pdfminer is primary — reliable, no ML, extracts all pages from text PDFs.
        # Docling (ML-based, better table/heading extraction) is reserved for when
        # the system has enough RAM (>8 GB free) or GPU. Re-enable by swapping the
        # call order below and ensuring Developer Mode is on for HF symlink support.
        try:
            doc = self._parse_with_pdfminer(path)
        except Exception as exc:
            logger.warning(f"pdfminer failed on {path.name}: {exc}. Falling back to pypdf.")
            doc = self._parse_with_pypdf(path)

        logger.info(f"  -> {len(doc.pages)} pages extracted")
        return doc

    # ------------------------------------------------------------------
    # pdfminer (primary) — reliable plain text, all pages, no ML
    # ------------------------------------------------------------------
    def _parse_with_pdfminer(self, path: Path) -> ParsedDocument:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer

        pages = []
        for page_no, page_layout in enumerate(extract_pages(str(path)), start=1):
            parts = []
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    text = element.get_text().strip()
                    if text:
                        parts.append(text)
            page_text = "\n".join(parts).strip()
            if page_text:
                pages.append(ParsedPage(page_no=page_no, text=page_text))

        return ParsedDocument(
            source_path=str(path),
            doc_type="pdf",
            pages=pages,
            title=path.stem,
            total_pages=len(pages),
        )

    # ------------------------------------------------------------------
    # pypdf (fallback) — lightweight, pure Python
    # ------------------------------------------------------------------
    def _parse_with_pypdf(self, path: Path) -> ParsedDocument:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(ParsedPage(page_no=page_no, text=text.strip()))

        return ParsedDocument(
            source_path=str(path),
            doc_type="pdf",
            pages=pages,
            title=path.stem,
            total_pages=len(pages),
        )

    # ------------------------------------------------------------------
    # Docling (optional, disabled) — ML-based, better table + heading
    # extraction. Re-enable when running on machine with >8 GB free RAM.
    # Set do_table_structure=True for full table extraction.
    # ------------------------------------------------------------------
    def _parse_with_docling(self, path: Path) -> ParsedDocument:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        pipeline_opts = PdfPipelineOptions()
        pipeline_opts.do_ocr = False
        pipeline_opts.do_table_structure = False

        converter = DocumentConverter(
            format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_opts)}
        )
        result = converter.convert(str(path))
        doc = result.document

        page_texts: dict[int, list[str]] = {}
        page_headings: dict[int, str] = {}
        current_heading = ""

        for item, _ in doc.iterate_items():
            page_no = 1
            if hasattr(item, "prov") and item.prov:
                page_no = item.prov[0].page_no

            label = str(getattr(item, "label", "")).lower()

            if "section_header" in label or "heading" in label:
                text = getattr(item, "text", "")
                current_heading = text
                page_headings[page_no] = text
                entry = f"## {text}"
            elif "table" in label:
                try:
                    entry = item.export_to_markdown(doc)
                except Exception:
                    entry = getattr(item, "text", "")
            else:
                entry = getattr(item, "text", "")

            if entry and entry.strip():
                page_texts.setdefault(page_no, []).append(entry)
                if page_no not in page_headings:
                    page_headings[page_no] = current_heading

        pages = [
            ParsedPage(
                page_no=pno,
                text="\n\n".join(parts),
                section_heading=page_headings.get(pno, ""),
            )
            for pno, parts in sorted(page_texts.items())
        ]

        return ParsedDocument(
            source_path=str(path),
            doc_type="pdf",
            pages=pages,
            title=getattr(doc, "name", None) or path.stem,
            total_pages=len(pages),
        )
