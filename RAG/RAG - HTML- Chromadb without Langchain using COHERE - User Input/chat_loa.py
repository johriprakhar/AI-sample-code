"""
HTML RAG pipeline: ChromaDB vector store + Cohere chat, without LangChain chains.

Configuration is loaded from config.py; secrets should be stored in .env.
Usage:
    python chat_loa.py build
    python chat_loa.py ask "your question"
    python chat_loa.py chat
    python chat_loa.py chat --rebuild
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import chromadb
import numpy as np
from cohere import ClientV2
from langchain_community.document_loaders import BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import config


COLLECTION_NAME = "html_rag"
_embedding_model: SentenceTransformer | None = None


# --------------------------------------------------------------------------- #
# Embedding model
# --------------------------------------------------------------------------- #

def get_embedding_model() -> SentenceTransformer:
    """Load the sentence-transformer model once and reuse it."""
    global _embedding_model

    if _embedding_model is None:
        print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    return _embedding_model


# --------------------------------------------------------------------------- #
# HTML loading and chunking
# --------------------------------------------------------------------------- #

def chunk_html() -> list[str]:
    """Load the configured HTML file and split its text into overlapping chunks."""
    if not config.HTML_PATH.exists():
        raise FileNotFoundError(f"HTML not found: {config.HTML_PATH}")

    if config.CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be greater than zero.")
    if config.CHUNK_OVERLAP < 0 or config.CHUNK_OVERLAP >= config.CHUNK_SIZE:
        raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE.")

    loader = BSHTMLLoader(str(config.HTML_PATH), open_encoding="utf-8")
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    chunks = [
        doc.page_content.strip()
        for page in pages
        for doc in splitter.create_documents([page.page_content])
        if doc.page_content.strip()
    ]

    if not chunks:
        raise ValueError(f"No extractable text found in {config.HTML_PATH}")

    print(f"Loaded {len(pages)} page(s), produced {len(chunks)} chunk(s).")
    return chunks


# --------------------------------------------------------------------------- #
# ChromaDB indexing
# --------------------------------------------------------------------------- #

def get_chroma_client() -> chromadb.PersistentClient:
    """Create a persistent ChromaDB client at the configured storage path."""
    config.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.VECTOR_STORE_DIR))


def build_index() -> None:
    """Embed HTML chunks and persist them in ChromaDB."""
    chunks = chunk_html()
    model = get_embedding_model()

    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError("Embedding model returned an unexpected embedding shape.")

    client = get_chroma_client()

    # Delete only this application's collection, then recreate it to avoid
    # stale chunks after the source HTML changes.
    existing_names = {
        item if isinstance(item, str) else item.name
        for item in client.list_collections()
    }
    if COLLECTION_NAME in existing_names:
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "l2"},
    )

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[
            {"source": str(config.DOCUMENT_SOURCE)}
            for _ in chunks
        ],
    )

    print(
        "ChromaDB index created successfully.\n"
        f"Vectors stored: {collection.count()}\n"
        f"Embedding dimension: {embeddings.shape[1]}\n"
        f"Persistent directory: {config.VECTOR_STORE_DIR}"
    )


# --------------------------------------------------------------------------- #
# Load and query ChromaDB
# --------------------------------------------------------------------------- #

def load_store() -> chromadb.Collection:
    """Load the persisted ChromaDB collection."""
    if not config.VECTOR_STORE_DIR.exists():
        raise FileNotFoundError(
            f"Vector store directory not found: {config.VECTOR_STORE_DIR}. "
            "Run `python chat_loa.py build` first."
        )

    client = get_chroma_client()

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        raise FileNotFoundError(
            f"ChromaDB collection '{COLLECTION_NAME}' was not found. "
            "Run `python chat_loa.py build` first."
        ) from exc

    if collection.count() == 0:
        raise ValueError(
            "The ChromaDB collection is empty. Run `python chat_loa.py build` again."
        )

    return collection


def retrieve(
    query: str,
    collection: chromadb.Collection,
) -> tuple[list[str], list[str], np.ndarray]:
    """Return top-k chunk texts, source labels, and L2 distances."""
    if not query.strip():
        return [], [], np.empty((1, 0), dtype=np.float32)

    count = collection.count()
    if count == 0:
        return [], [], np.empty((1, 0), dtype=np.float32)

    model = get_embedding_model()
    query_embedding = np.asarray(
        model.encode(query, convert_to_numpy=True),
        dtype=np.float32,
    ).reshape(-1)

    k = min(int(config.TOP_K), count)
    if k <= 0:
        raise ValueError("TOP_K must be greater than zero.")

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    texts = results["documents"][0] or []
    metadatas = results["metadatas"][0] or []
    raw_distances = results["distances"][0] or []

    sources = [
        metadata.get("source", "")
        for metadata in metadatas
        if metadata is not None
    ]
    distances = np.asarray(raw_distances, dtype=np.float32).reshape(1, -1)

    return texts, sources, distances


# --------------------------------------------------------------------------- #
# Cohere answer generation and query handling
# --------------------------------------------------------------------------- #

def ask_cohere(client: ClientV2, context: str, question: str) -> str:
    """Send retrieved context and the question to Cohere."""
    prompt = config.PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )

    response = client.chat(
        model=config.COHERE_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.COHERE_TEMPERATURE,
    )

    # ClientV2 normally returns text blocks in message.content.
    content = response.message.content or []
    answer_parts = [
        block.text
        for block in content
        if getattr(block, "text", None)
    ]
    if not answer_parts:
        raise RuntimeError("Cohere returned no text content.")
    return "\n".join(answer_parts).strip()


def answer_query(
    query: str,
    collection: chromadb.Collection,
    client: ClientV2,
) -> dict[str, Any]:
    """Retrieve context, apply the configured L2 distance gate, and answer."""
    texts, sources, distances = retrieve(query, collection)
    scores = [float(distance) for distance in distances[0]]
    best_score = scores[0] if scores else float("inf")

    result: dict[str, Any] = {
        "question": query,
        "answered": False,
        "answer": config.IRRELEVANT_QUERY_MESSAGE,
        "sources": [],
        "scores": scores,
        "best_score": best_score,
        "threshold": config.DISTANCE_THRESHOLD,
        "chunks": [],
    }

    # ChromaDB L2 distances are distances, not confidence percentages.
    if not texts or best_score > config.DISTANCE_THRESHOLD:
        return result

    result["answer"] = ask_cohere(client, "\n\n".join(texts), query)
    result["answered"] = True
    result["sources"] = sorted({source for source in sources if source})
    result["chunks"] = texts
    return result


def print_result(result: dict[str, Any]) -> None:
    """Print the structured result in the command-line interface."""
    print("L2 distance scores:", result["scores"])

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
    """Run a single query or start an interactive chat session."""
    config.validate()
    collection = load_store()
    client = ClientV2(api_key=config.COHERE_API_KEY)

    if single_query is not None:
        print_result(answer_query(single_query, collection, client))
        return

    print("Ask a question about the HTML document. Type 'exit' or 'quit' to stop.")
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

        try:
            print_result(answer_query(query, collection, client))
        except Exception as exc:
            print(f"Error processing query: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="HTML RAG with ChromaDB + Cohere (settings in config.py / .env)"
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "build",
        help="Build the ChromaDB index from the configured HTML",
    )

    ask_parser = subparsers.add_parser("ask", help="Answer a single question")
    ask_parser.add_argument("question", help="The question to answer")
    ask_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the index before answering",
    )

    chat_parser = subparsers.add_parser("chat", help="Interactive question loop")
    chat_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the index before chatting",
    )

    args = parser.parse_args(argv)
    command = args.command or "chat"

    try:
        if command == "build":
            build_index()
            return 0

        if getattr(args, "rebuild", False):
            config.validate()
            build_index()

        run_chat(args.question if command == "ask" else None)
        return 0

    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
