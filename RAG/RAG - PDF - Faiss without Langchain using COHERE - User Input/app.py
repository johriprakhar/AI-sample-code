"""Flask web UI for the PDF RAG pipeline.

Serves a single page where you type a question, see a loader while the pipeline
runs, and get the answer plus its sources and retrieval scores rendered as HTML.

The retrieval and Cohere logic is not duplicated here — this module is a thin
HTTP layer over `chat_loa.answer_query`.

Usage:
    python app.py                      # http://127.0.0.1:5000
    python app.py --port 8000
    python app.py --host 0.0.0.0       # exposes the app on your network

SECURITY: there is no authentication on these endpoints. Every visitor can
spend your Cohere credits and read the indexed document. Keep the default
127.0.0.1 binding unless you have put your own auth/proxy in front of it, and
never run this with `--host 0.0.0.0` on an untrusted network.

Author:
    Prakhar Johri — https://www.linkedin.com/in/johriprakhar/
"""

from __future__ import annotations

import argparse
import threading
from typing import Any, Dict, Tuple

from flask import Flask, jsonify, render_template, request

import chat_loa
import config

# Longest question we accept. Keeps prompt cost and payload size bounded.
MAX_QUESTION_LENGTH = 1000

app = Flask(__name__)

# Reject oversized request bodies before they are parsed.
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

# --------------------------------------------------------------------------- #
# Lazily-initialised pipeline state
# --------------------------------------------------------------------------- #
# The embedding model, FAISS index, chunk table and Cohere client are built once
# on the first request and reused. The lock serialises both the initialisation
# and the per-query work, since neither the sentence-transformer encoder nor the
# Cohere client is guaranteed thread-safe and Flask's dev server is threaded.
_lock = threading.Lock()
_pipeline: Dict[str, Any] | None = None


class PipelineUnavailable(RuntimeError):
    """Raised when the pipeline cannot be initialised (bad config, no index)."""


def get_pipeline() -> Dict[str, Any]:
    """Return the initialised pipeline, building it on first use."""
    global _pipeline
    if _pipeline is None:
        try:
            config.validate()
            index, df = chat_loa.load_store()
            chat_loa.get_embedding_model()
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            raise PipelineUnavailable(str(exc)) from exc

        _pipeline = {
            "index": index,
            "df": df,
            "client": chat_loa.ClientV2(api_key=config.COHERE_API_KEY),
        }
    return _pipeline


def settings_summary(vectors: int | None = None) -> Dict[str, Any]:
    """Config values worth showing in the UI footer / status panel."""
    return {
        "pdf": config.PDF_PATH.name,
        "document_source": config.DOCUMENT_SOURCE,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "chat_model": config.COHERE_CHAT_MODEL,
        "top_k": config.TOP_K,
        "threshold": config.DISTANCE_THRESHOLD,
        "vectors": vectors,
    }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def index() -> str:
    """The search page. Renders even when the index is missing, so the user
    sees the reason in the UI instead of a stack trace."""
    return render_template("index.html", settings=settings_summary())


@app.get("/api/status")
def api_status() -> Tuple[Any, int]:
    """Report whether the pipeline is usable, and with what settings."""
    with _lock:
        try:
            pipeline = get_pipeline()
        except PipelineUnavailable as exc:
            return jsonify({"ready": False, "error": str(exc),
                            "settings": settings_summary()}), 503
        vectors = pipeline["index"].ntotal

    return jsonify({"ready": True, "settings": settings_summary(vectors)}), 200


@app.post("/api/ask")
def api_ask() -> Tuple[Any, int]:
    """Answer one question. Returns the `chat_loa.answer_query` result as JSON."""
    payload = request.get_json(silent=True) or {}
    question = payload.get("question")

    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Please enter a question."}), 400

    question = question.strip()
    if len(question) > MAX_QUESTION_LENGTH:
        return jsonify(
            {"error": f"Question is too long (limit {MAX_QUESTION_LENGTH} characters)."}
        ), 400

    with _lock:
        try:
            pipeline = get_pipeline()
        except PipelineUnavailable as exc:
            return jsonify({"error": str(exc)}), 503

        try:
            result = chat_loa.answer_query(
                question, pipeline["index"], pipeline["df"], pipeline["client"]
            )
        except Exception as exc:  # noqa: BLE001 - surface a clean message, log the rest
            app.logger.exception("Query failed")
            return jsonify(
                {"error": f"The query failed: {type(exc).__name__}: {exc}"}
            ), 502

    return jsonify(result), 200


@app.errorhandler(404)
def not_found(_exc: Any) -> Tuple[Any, int]:
    return jsonify({"error": "Not found."}), 404


@app.errorhandler(413)
def too_large(_exc: Any) -> Tuple[Any, int]:
    return jsonify({"error": "Request body too large."}), 413


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Web UI for the PDF RAG pipeline")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind. Default 127.0.0.1 (this machine only).",
    )
    parser.add_argument("--port", type=int, default=5000, help="Port. Default 5000.")
    parser.add_argument(
        "--debug", action="store_true", help="Enable the Flask reloader and debugger."
    )
    args = parser.parse_args()

    if args.host not in ("127.0.0.1", "localhost"):
        print(
            f"WARNING: binding to {args.host} exposes this unauthenticated app "
            "to other machines on the network."
        )

    print(f"PDF RAG web UI on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
