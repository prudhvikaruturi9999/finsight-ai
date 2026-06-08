#!/usr/bin/env python3
"""FinSight AI — Phase 1 ingestion pipeline entry point.

Usage:
    python ingest.py                           # ingest all PDFs in tests/fixtures
    python ingest.py --source path/to/dir      # custom directory
    python ingest.py --file path/to/file.pdf   # single file
    python ingest.py --child-only              # embed only child chunks (faster/cheaper)
    python ingest.py --reset                   # delete existing vectors for re-indexed files
"""
import sys
from pathlib import Path

import click

from config import settings
from processors.pdf_processor import PdfProcessor
from chunking.router import chunk_document
from embedding.embedder import Embedder
from vector_store.chroma_store import ChromaStore
from utils.logger import get_logger

logger = get_logger("ingest")

SUPPORTED_EXTENSIONS = {".pdf"}


@click.command()
@click.option("--source", default="tests/fixtures", show_default=True,
              help="Directory of documents to ingest.")
@click.option("--file", "single_file", default=None,
              help="Ingest a single file instead of a directory.")
@click.option("--child-only", is_flag=True, default=False,
              help="Only embed child chunks. Parent chunks stored as metadata only.")
@click.option("--reset", is_flag=True, default=False,
              help="Delete existing vectors for each file before re-indexing.")
def main(source: str, single_file: str | None, child_only: bool, reset: bool) -> None:
    """FinSight AI — document ingestion pipeline (Phase 1: PDFs)."""

    # ── Collect files ─────────────────────────────────────────────────
    if single_file:
        files = [Path(single_file)]
    else:
        source_dir = Path(source)
        if not source_dir.exists():
            logger.error(f"Source directory not found: {source_dir.resolve()}")
            sys.exit(1)
        files = sorted(
            f for f in source_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS and not f.name.startswith(".")
        )

    # Deduplicate by canonical path (e.g. remove duplicate "GOOG-10-K-2025 (1).pdf")
    seen_names: set[str] = set()
    unique_files = []
    for f in files:
        if f.name not in seen_names:
            seen_names.add(f.name)
            unique_files.append(f)
        else:
            logger.warning(f"Skipping duplicate filename: {f.name}")
    files = unique_files

    if not files:
        logger.warning("No supported files found — nothing to do.")
        return

    logger.info(f"Files to process: {len(files)}")
    for f in files:
        logger.info(f"  {f.name}")

    # ── Init pipeline components ──────────────────────────────────────
    pdf_processor = PdfProcessor()
    embedder = Embedder()
    store = ChromaStore()

    # ── Process each file ─────────────────────────────────────────────
    total_new = 0

    for file_path in files:
        logger.info(f"\n{'-' * 60}")
        logger.info(f"Processing: {file_path.name}")

        if reset:
            deleted = store.delete_by_source(str(file_path))
            if deleted:
                logger.info(f"  Deleted {deleted} existing vectors for this file.")

        # 1. Parse
        doc = pdf_processor.parse(str(file_path))
        if not doc.pages:
            logger.warning(f"  No content extracted — skipping.")
            continue
        logger.info(f"  Parsed {len(doc.pages)} pages  (title: {doc.title!r})")

        # 2. Chunk
        chunks = chunk_document(doc)
        if child_only:
            chunks = [c for c in chunks if c.metadata.chunk_type == "child"]

        # 3. Embed (cached chunks skipped automatically)
        chunks_with_embeddings = embedder.embed_chunks(chunks)

        # 4. Store
        n = store.upsert(chunks_with_embeddings)
        total_new += n
        logger.info(f"  {n} new chunks indexed for {file_path.name}")

    # ── Summary ───────────────────────────────────────────────────────
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Ingestion complete.")
    logger.info(f"  New chunks indexed : {total_new}")
    logger.info(f"  Total in store     : {store.count()}")
    logger.info(f"  Embedding stats    : {embedder.cost_tracker.summary()}")


if __name__ == "__main__":
    main()
