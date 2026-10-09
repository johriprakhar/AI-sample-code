"""PDF RAG pipeline: FAISS vector store + Cohere chat, no LangChain chains.

Script version of chat_loa_notebook.ipynb. All settings live in config.py,
secrets in .env.

Usage:
    python chat_loa.py build                 # (re)build the FAISS index from the PDF
    python chat_loa.py ask "your question"   # single question
    python chat_loa.py chat                  # interactive loop
    python chat_loa.py chat --rebuild        # rebuild first, then chat

Author:
    Prakhar Johri — https://www.linkedin.com/in/johriprakhar/
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

import faiss
import numpy as np
import pandas as pd
from cohere import ClientV2
#from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import BSHTMLLoader


import config


# --------------------------------------------------------------------------- #
# Embedding model
# --------------------------------------------------------------------------- #
_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Load the sentence-transformer model once and reuse it."""
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _embedding_model


# --------------------------------------------------------------------------- #
# Index building
# --------------------------------------------------------------------------- #
def chunk_html() -> List[str]:
    """Load the configured HTML and split each page into overlapping chunks."""
    if not config.HTML_PATH.exists():
        raise FileNotFoundError(f"HTML not found: {config.HTML_PATH}")

    loader = BSHTMLLoader(str(config.HTML_PATH),open_encoding="utf-8")
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    chunks: List[str] = []
    for page in pages:  # one page at a time from the HTML
        for doc in splitter.create_documents([page.page_content]):
            chunks.append(doc.page_content)

    if not chunks:
        raise ValueError(f"No extractable text found in {config.HTML_PATH}")

    print(f"Loaded {len(pages)} page(s), produced {len(chunks)} chunk(s).")
    return chunks


def build_index() -> None:
    """Embed the HTML chunks, build a FAISS index, and persist it with the chunks."""
    chunks = chunk_html()
    model = get_embedding_model()

    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    config.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {"documents": chunks, "source": [config.DOCUMENT_SOURCE] * len(chunks)}
    ).to_csv(config.DOCS_CSV_PATH, index=False)

    faiss.write_index(index, str(config.INDEX_PATH))

    print(
        f"FAISS index ({index.ntotal} vectors, dim {embeddings.shape[1]}) written to "
        f"{config.INDEX_PATH}\nDocument store written to {config.DOCS_CSV_PATH}"
    )


# --------------------------------------------------------------------------- #
# Retrieval + answering
# --------------------------------------------------------------------------- #
def load_store() -> Tuple[faiss.Index, pd.DataFrame]:
    """Read the persisted FAISS index and the matching chunk table."""
    if not config.INDEX_PATH.exists() or not config.DOCS_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Vector store missing in {config.VECTOR_STORE_DIR}. "
            "Run `python chat_loa.py build` first."
        )
    return faiss.read_index(str(config.INDEX_PATH)), pd.read_csv(config.DOCS_CSV_PATH)


def retrieve(
    query: str, index: faiss.Index, df: pd.DataFrame
) -> Tuple[List[str], List[str], np.ndarray]:
    """Return the top-k chunk texts, their sources, and the distance array."""
    model = get_embedding_model()
    query_embedding = np.asarray(
        model.encode(query), dtype="float32"
    ).reshape(1, -1)

    k = min(config.TOP_K, index.ntotal)
    distances, indices = index.search(query_embedding, k)

    texts = [df.loc[i, "documents"] for i in indices[0] if i >= 0]
    sources = [df.loc[i, "source"] for i in indices[0] if i >= 0]
    return texts, sources, distances


def ask_cohere(client: ClientV2, context: str, question: str) -> str:
    """Send the retrieved context plus the question to Cohere and return the answer."""
    prompt = config.PROMPT_TEMPLATE.format(context=context, question=question)

    response = client.chat(
        model=config.COHERE_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.COHERE_TEMPERATURE,
    )
    return response.message.content[0].text


def answer_query(
    query: str, index: faiss.Index, df: pd.DataFrame, client: ClientV2
) -> dict:
    """Retrieve, gate on distance, and return a structured result.

    The returned dict is the shared contract for both the CLI and the web UI:

        question   the query as asked
        answered   False when the distance gate rejected the query
        answer     Cohere's answer, or IRRELEVANT_QUERY_MESSAGE
        sources    sorted unique source labels (empty when rejected)
        scores     L2 distance for each retrieved chunk, nearest first
        best_score scores[0], the value compared against the threshold
        threshold  DISTANCE_THRESHOLD in effect for this call
        chunks     the retrieved chunk texts (empty when rejected)
    """
    texts, sources, distances = retrieve(query, index, df)
    scores = [float(d) for d in distances[0]] if distances.size else []
    best_score = scores[0] if scores else float("inf")

    result = {
        "question": query,
        "answered": False,
        "answer": config.IRRELEVANT_QUERY_MESSAGE,
        "sources": [],
        "scores": scores,
        "best_score": best_score,
        "threshold": config.DISTANCE_THRESHOLD,
        "chunks": [],
    }

    if best_score > config.DISTANCE_THRESHOLD:
        return result

    result["answered"] = True
    result["answer"] = ask_cohere(client, " ".join(texts), query)
    result["sources"] = sorted(set(sources))
    result["chunks"] = texts
    return result


def print_result(result: dict) -> None:
    """Render an answer_query result the way the CLI has always printed it."""
    print("Distance score:", [[*result["scores"]]])

    if not result["answered"]:
        print(result["answer"])
        return

    print("\nBot Response:")
    print(result["answer"])
    print("\nSources:")
    print(result["sources"])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run_chat(single_query: str | None = None) -> None:
    config.validate()
    index, df = load_store()
    client = ClientV2(api_key=config.COHERE_API_KEY)

    if single_query is not None:
        print_result(answer_query(single_query, index, df, client))
        return

    print("Ask a question about the document. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            query = input("\nEnter your query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        print_result(answer_query(query, index, df, client))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="HTML RAG with FAISS + Cohere (settings in config.py / .env)"
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("build", help="Build the FAISS index from the configured HTML")

    ask_parser = subparsers.add_parser("ask", help="Answer a single question")
    ask_parser.add_argument("question", help="The question to answer")
    ask_parser.add_argument(
        "--rebuild", action="store_true", help="Rebuild the index before answering"
    )

    chat_parser = subparsers.add_parser("chat", help="Interactive question loop")
    chat_parser.add_argument(
        "--rebuild", action="store_true", help="Rebuild the index before chatting"
    )

    args = parser.parse_args(argv)
    command = args.command or "chat"

    try:
        if command == "build":
            build_index()
            return 0

        if getattr(args, "rebuild", False):
            build_index()

        run_chat(args.question if command == "ask" else None)
        return 0
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
