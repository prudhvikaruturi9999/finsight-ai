# FinSight AI — Enterprise RAG Ingestion Pipeline

A production-grade, multi-source RAG (Retrieval-Augmented Generation) ingestion pipeline that continuously collects, processes, and indexes heterogeneous enterprise data into a vector store for accurate, cited AI-powered search.

Built as a learning project covering every major real-world ingestion pattern used in production AI systems.

---

## What this does

Takes raw data from 10+ source types, converts everything into vector embeddings, and stores them in a searchable vector database. A retrieval layer then surfaces the most relevant chunks to an LLM to answer natural-language questions with citations.

**Example queries this system can answer:**
- *"What did the CFO say about margins in the last earnings call?"*
- *"Summarise all compliance policy changes from this quarter"*
- *"What are competitors doing in embedded finance?"*

---

## Data sources

| Source | Format | Ingestion pattern |
|---|---|---|
| News sites, regulatory portals | HTML | Web scraping (Scrapy + Playwright) |
| Financial data, internal APIs | JSON | REST / GraphQL with pagination |
| Annual reports, research reports | PDF | Docling + OCR fallback |
| Strategy docs, board decks | DOCX / PPTX | Unstructured element extraction |
| Financial models, market data | Excel / CSV | Row-level chunking with header context |
| Document archive, media files | Any | AWS S3 event-driven + batch backfill |
| Transaction records, metadata | Rows | PostgreSQL incremental sync (CDC) |
| Earnings calls, webinars | Audio / Video | Whisper transcription + diarization |
| Scanned contracts, chart images | Images | Tesseract OCR + VLM captioning |
| Internal discussions | Slack / Email | Thread reconstruction + PII scrubbing |

---

## Architecture

```
Raw sources
    │
    ▼
┌─────────────────────────────────────────┐
│           Source connectors             │
│  Web · API · PDF · DOCX · Excel · AWS  │
│       Audio · Images · Messaging        │
└──────────────────┬──────────────────────┘
                   │  normalised JSON/Markdown
                   ▼
┌─────────────────────────────────────────┐
│         Document processors             │
│   Parse · OCR · Transcribe · Caption   │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│      Enrichment + chunking              │
│  Metadata · NER tagging · Permissions  │
│  Hierarchical / semantic / row chunks  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│         Embedding service               │
│   OpenAI / Cohere / BGE · Dedup cache  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│           Vector store                  │
│     Qdrant · HNSW + BM25 hybrid index  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│          Retrieval layer                │
│  Hybrid search · Reranking · Filters   │
└─────────────────────────────────────────┘
```

All stages are orchestrated by Airflow with per-source DAGs, dead-letter queues, and Prometheus + Grafana monitoring.

---

## Tech stack

**Parsing & extraction**
- [Docling](https://github.com/DS4SD/docling) — PDF layout analysis and table extraction
- [Unstructured](https://unstructured.io) — DOCX, PPTX, HTML semantic element extraction
- [Whisper](https://github.com/openai/whisper) — audio transcription
- [Tesseract](https://github.com/tesseract-ocr/tesseract) — OCR for scanned documents
- GPT-4o Vision — chart and diagram captioning
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — speaker diarization

**Chunking & embedding**
- [LlamaIndex](https://www.llamaindex.ai) — hierarchical and semantic chunking
- OpenAI `text-embedding-3-large` / Cohere `embed-v3` / BGE (local fallback)
- SHA-256 content-hash dedup cache (SQLite-backed)

**Vector store**
- [Qdrant](https://qdrant.tech) — production (HNSW + BM25 hybrid search)
- [Chroma](https://www.trychroma.com) — local development

**Retrieval**
- Reciprocal Rank Fusion (RRF) for hybrid search result fusion
- [Cohere Rerank](https://cohere.com/rerank) / BGE cross-encoder reranking

**Cloud & infra**
- AWS S3, SQS, Lambda, RDS (PostgreSQL)
- [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) for all AWS operations

**Orchestration & monitoring**
- [Apache Airflow](https://airflow.apache.org) — DAG scheduling and pipeline management
- [Prometheus](https://prometheus.io) + [Grafana](https://grafana.com) — metrics and dashboards
- [Microsoft Presidio](https://microsoft.github.io/presidio/) — PII detection and redaction

---

## Project structure

```
finsight-ai/
├── connectors/          # One folder per source type
│   ├── web/             # Scrapy + Playwright scraper
│   ├── api/             # REST / GraphQL connector
│   ├── aws/             # S3 + RDS connector
│   └── messaging/       # Slack / Teams / Email + PII scrubber
├── processors/          # Per-modality document parsers
├── chunking/            # Chunking strategies (hierarchical, semantic, row-level)
├── enrichment/          # Metadata schema, NER tagging, permission mapping
├── embedding/           # Embedding service, dedup cache, cost tracker
├── vector_store/        # Vector DB adapters (Qdrant, Chroma, Weaviate, Pinecone)
├── retrieval/           # Hybrid search, reranker, permission filter
├── orchestration/       # Airflow DAGs, DLQ handler, lineage logger
├── monitoring/          # Prometheus metrics, alerting, Grafana dashboard
├── evaluation/          # Recall@10 eval harness, PII scanner, golden eval set
├── utils/               # Logger, secrets loader, retry decorator
├── config/              # Settings, source definitions, chunking config
├── tests/               # Unit + integration tests with fixtures
└── docs/                # Architecture notes, connector guide, runbooks
```

---

## Build phases

This project is built incrementally across 8 phases (~185 hours total at 5 hrs/day):

| Phase | What gets built | Target |
|---|---|---|
| 1 | PDF + DOCX + Chroma — end-to-end RAG working | Week 2 |
| 2 | Web scraping + REST/GraphQL API connectors | Week 3 |
| 3 | Excel, image OCR/captioning, audio transcription | Week 4 |
| 4 | AWS S3 event-driven + RDS incremental sync | Week 6 |
| 5 | Slack/Teams/email + PII scrubbing + permission scoping | Week 7 |
| 6 | Migrate to Qdrant, hybrid search, reranking | Week 7 |
| 7 | Airflow orchestration, DLQ, Prometheus + Grafana | Week 8 |
| 8 | Security hardening, cost controls, eval harness | Week 8 |

---

## Getting started

**Prerequisites**
- Python 3.11+
- Docker Desktop (for Qdrant, Airflow, Prometheus, Grafana)
- Git

**Setup**

```bash
git clone https://github.com/YOUR_USERNAME/finsight-ai.git
cd finsight-ai

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac / Linux

pip install -r requirements.txt

copy .env.example .env       # Windows
# cp .env.example .env       # Mac / Linux
# Fill in your API keys in .env

docker-compose up -d
```

**Run Phase 1 ingestion (PDF)**

```bash
python -m connectors.pdf_processor --source ./tests/fixtures/sample.pdf
```

---

## Key production features

- **Incremental sync** — only re-processes changed or new documents, not the full corpus
- **Content-hash dedup cache** — identical chunks are never re-embedded, cutting API costs significantly
- **Permission-aware indexing** — private Slack channels and confidential documents are access-scoped at the chunk level
- **Dead-letter queue** — failed documents are captured with full error context and can be replayed with one command
- **Per-document lineage logging** — every document's full journey through the pipeline is auditable
- **PII scrubbing before embedding** — personal data is redacted before vectors are created, not just before display
- **Hybrid search** — dense vector similarity + BM25 keyword search fused with RRF for better retrieval accuracy
- **Cross-encoder reranking** — top-20 candidates reranked before returning top-5 to the LLM

---

## Environment variables

See `.env.example` for the full list. Required keys to get started:

```
OPENAI_API_KEY        — for embeddings (Phase 1)
QDRANT_URL            — vector store (default: http://localhost:6333)
```

All other keys are needed progressively as you build each phase.

---

## Evaluation targets

| Metric | Target |
|---|---|
| Recall@10 on golden eval set | > 80% |
| Retrieval latency P95 | < 500 ms |
| DLQ failure rate per source | < 1% |
| PII entities in vector store | 0 |

---

## License

MIT