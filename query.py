#!/usr/bin/env python3
"""FinSight AI — RAG query interface (Phase 1: M1 demo).

Usage:
    python query.py "What was Google total revenue in 2025?"
    python query.py "What was the operating margin?" --top-k 5
    python query.py --interactive
"""
import sys
import click

from config import settings
from retrieval.retriever import Retriever
from utils.logger import get_logger

logger = get_logger("query")

_SYSTEM_PROMPT = """You are FinSight AI, a financial research assistant.
Answer questions using ONLY the provided source excerpts.
Always cite your sources using the [N] reference numbers given in the context.
If the answer is not found in the excerpts, say so clearly — do not speculate.
Be concise and precise. Use numbers exactly as they appear in the source."""

_USER_TEMPLATE = """\
Question: {question}

Source excerpts:
{context}

Answer the question concisely, citing the relevant [N] source numbers."""


def _call_llm(question: str, context: str) -> str:
    prompt = _USER_TEMPLATE.format(question=question, context=context)

    if settings.LLM_PROVIDER == "anthropic":
        return _call_anthropic(prompt)
    return _call_openai(prompt)


def _call_anthropic(prompt: str) -> str:
    if not settings.ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file.\n"
            "Get a key at: https://console.anthropic.com/"
        )
    import anthropic
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=settings.LLM_MODEL,
        max_tokens=settings.LLM_MAX_TOKENS,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _call_openai(prompt: str) -> str:
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in .env.")
    from openai import OpenAI
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=settings.LLM_MODEL if "gpt" in settings.LLM_MODEL else "gpt-4o-mini",
        max_tokens=settings.LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


def _answer(question: str, top_k: int, show_chunks: bool) -> None:
    retriever = Retriever()
    result = retriever.retrieve(question, top_k=top_k)

    if not result.chunks:
        print("No relevant content found in the knowledge base.")
        return

    context = result.format_context()

    if show_chunks:
        print("\n--- Retrieved chunks ---")
        for i, c in enumerate(result.chunks, 1):
            from pathlib import Path
            src = Path(c.source_file).name
            print(f"[{i}] {src} p.{c.page_no}  score={c.score:.3f}")
            print(f"     {c.text[:120].strip()!r}")
        print()

    try:
        answer = _call_llm(question, context)
    except ValueError as e:
        print(f"\nLLM not configured: {e}")
        print("\nRetrieved context (no LLM answer):")
        print(context[:2000])
        return

    print(f"\nAnswer:\n{answer}")
    print(f"\nSources:\n{result.format_citations()}")


@click.command()
@click.argument("question", required=False)
@click.option("--top-k", default=settings.RETRIEVAL_TOP_K, show_default=True,
              help="Number of chunks to retrieve.")
@click.option("--show-chunks", is_flag=True, default=False,
              help="Print retrieved chunks before the answer.")
@click.option("--interactive", "-i", is_flag=True, default=False,
              help="Start an interactive Q&A session.")
def main(question: str | None, top_k: int, show_chunks: bool, interactive: bool) -> None:
    """FinSight AI — ask questions against the indexed SEC filings."""

    if interactive:
        print("FinSight AI — Interactive mode  (type 'exit' to quit)\n")
        while True:
            try:
                q = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in {"exit", "quit", "q"}:
                break
            _answer(q, top_k, show_chunks)
            print()
        return

    if not question:
        click.echo(click.get_current_context().get_help())
        sys.exit(0)

    _answer(question, top_k, show_chunks)


if __name__ == "__main__":
    main()
