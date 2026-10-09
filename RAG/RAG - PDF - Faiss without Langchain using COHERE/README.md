# PDF RAG — FAISS + Cohere (no LangChain chains)

Ask questions about a PDF. The script chunks the PDF, embeds the chunks locally
with a sentence-transformer model, stores the vectors in a FAISS index, then
answers questions by retrieving the closest chunks and passing them to Cohere.

LangChain is used only for PDF loading and text splitting — there are no chains,
retrievers, or agents.

## Files

| File | Purpose |
| --- | --- |
| `chat_loa.py` | The pipeline and CLI. |
| `config.py` | Every tunable setting. Reads overrides from `.env`. |
| `.env.example` | Template for `.env`. Copy it, don't edit it in place. |
| `.env` | Your secrets and overrides. Git-ignored. You create this. |
| `requirements.txt` | Dependencies. |
| `Vector_Store/` | Generated index + chunk table. Git-ignored. |
| `chat_loa_notebook.ipynb` | The original notebook this was converted from. |

## Requirements

- Python 3.12 or newer (floor set by the pinned `numpy==2.5.3`; verified on 3.14)
- A Cohere API key — get one at [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys)
- Roughly 2 GB of disk for the install (`sentence-transformers` pulls in PyTorch)

## Setup

### 1. Install dependencies

```powershell
cd "c:\wamp64\www\AI-sample-code\RAG\RAG - PDF - Faiss without Langchain using COHERE"
python -m pip install -r requirements.txt
```

A virtual environment is recommended so PyTorch doesn't land in your global
site-packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks the activate script, allow it for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### 2. Create your `.env`

```powershell
Copy-Item .env.example .env
```

Open `.env` and set your key:

```ini
COHERE_API_KEY=your-real-key-here
```

`.env` is git-ignored. Never commit it, and don't put the key in `config.py`.

### 3. Copy in your PDF and point the config at it

Copy the PDF you want to query into the project folder (the same folder as
`chat_loa.py`):

```powershell
Copy-Item "C:\path\to\your\document.pdf" .
```

Then update the filename in `.env` so it matches exactly:

```ini
PDF_PATH=document.pdf
```

Notes on this step:

- The name is case-sensitive on some filesystems and must include the `.pdf`
  extension. A mismatch produces `Error: PDF not found: ...` with the full path
  it tried, so compare that against `Get-ChildItem *.pdf`.
- A relative name is resolved against the project folder. You can also leave the
  PDF where it is and give an absolute path instead:
  `PDF_PATH=C:\Users\you\Documents\document.pdf`
- If the path has spaces, write it bare — no quotes: `PDF_PATH=My Report.pdf`
- The default is `Rhea_resume.pdf`, which is not included in this repo. You must
  supply your own PDF.
- Also update `DOCUMENT_SOURCE` in `.env` — that string is what gets printed
  under "Sources" for every answer, so set it to your document's origin
  (a URL, a filename, a system name, whatever is meaningful).
- The PDF must contain a real text layer. Scanned or image-only PDFs extract
  nothing and fail with `No extractable text found`. Run them through OCR first.

### 4. Build the index

```powershell
python chat_loa.py build
```

First run downloads the embedding model (~90 MB for `all-MiniLM-L6-v2`), so give
it a minute. On success you get the vector count and the two output paths. This
writes `Vector_Store/vector_db.index` and `Vector_Store/docs.csv`.

### 5. Ask questions

```powershell
python chat_loa.py chat
```

## Usage

```
python chat_loa.py build                        # (re)build the index from the configured PDF
python chat_loa.py ask "what are her skills?"   # one question, then exit
python chat_loa.py chat                         # interactive loop
python chat_loa.py chat --rebuild               # rebuild, then chat
python chat_loa.py ask "..." --rebuild          # rebuild, then answer
python chat_loa.py --help                       # full CLI help
```

Running with no subcommand defaults to `chat`. In the interactive loop, type
`exit` or `quit` (or press Ctrl+C) to leave.

### Swapping to a different PDF

Re-run steps 3 and 4 — copy the new file in, change `PDF_PATH`, then rebuild.
The rebuild overwrites the previous index, so one PDF is active at a time. The
shortcut is `python chat_loa.py chat --rebuild` after editing `.env`.

Forgetting to rebuild is the most common mistake: you'll get confident answers
sourced from the *old* document.

## Sample output

```
Enter your query: what are her skills?
Distance score: [[0.78 0.91 1.02 1.11 1.24]]

Bot Response:
Based on the document, her skills include ...

Sources:
['www.rheadata.com']
```

If the closest chunk is farther than `DISTANCE_THRESHOLD`, no Cohere call is
made and you get `Please ask a relevant question.` instead. That's the relevance
gate working, not an error.

## Configuration reference

Every value below can be set in `.env`. Defaults live in `config.py` and are
used when a variable is absent or blank.

| Variable | Default | What it does |
| --- | --- | --- |
| `COHERE_API_KEY` | *(none — required)* | Your Cohere key. The script exits with a clear message if unset. |
| `PDF_PATH` | `Rhea_resume.pdf` | PDF to index. Relative to the project folder, or absolute. |
| `DOCUMENT_SOURCE` | `www.rheadata.com` | Citation label attached to every chunk. |
| `VECTOR_STORE_DIR` | `Vector_Store` | Where the index and chunk table are written. |
| `INDEX_FILENAME` | `vector_db.index` | FAISS index filename. |
| `DOCS_FILENAME` | `docs.csv` | Chunk table filename. |
| `CHUNK_SIZE` | `500` | Characters per chunk. Larger means more context per hit, fewer hits. |
| `CHUNK_OVERLAP` | `100` | Character overlap between neighbouring chunks, so sentences aren't cut mid-thought. |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Any [sentence-transformers](https://www.sbert.net/docs/pretrained_models.html) model. Changing it requires a rebuild. |
| `TOP_K` | `5` | Chunks retrieved per question. Clamped down if the index holds fewer. |
| `DISTANCE_THRESHOLD` | `1.7` | Max L2 distance for the nearest chunk. Lower is stricter. |
| `COHERE_CHAT_MODEL` | `command-a-03-2025` | Cohere chat model. |
| `COHERE_TEMPERATURE` | `0.3` | Higher is more creative, lower is more literal. |
| `PROMPT_TEMPLATE` | see `config.py` | Must keep the `{context}` and `{question}` placeholders. |
| `IRRELEVANT_QUERY_MESSAGE` | `Please ask a relevant question.` | Shown when the threshold rejects a question. |

Changing `CHUNK_SIZE`, `CHUNK_OVERLAP`, or `EMBEDDING_MODEL_NAME` invalidates the
existing index — rebuild after editing any of them. The retrieval and Cohere
settings take effect immediately, no rebuild needed.

## Troubleshooting

**`Error: COHERE_API_KEY is not set.`**
No `.env`, or the key line is missing or blank. Confirm the file is named exactly
`.env` (Windows Explorer likes to append `.txt`) and sits beside `chat_loa.py`.
Check with `Get-ChildItem -Force .env`.

**`Error: PDF not found: <path>`**
`PDF_PATH` doesn't match a real file. The message shows the exact path tried —
compare it to `Get-ChildItem *.pdf`. Watch for a missing `.pdf` extension or a
typo in the name.

**`Error: No extractable text found in <path>`**
The PDF has no text layer, i.e. it's a scan or export of images. OCR it first
(Acrobat, `ocrmypdf`, or similar) and rebuild.

**``Error: Vector store missing in <dir>. Run `python chat_loa.py build` first.``**
You went straight to `ask`/`chat`. Run `build` once.

**Every question returns "Please ask a relevant question."**
The nearest chunk exceeds `DISTANCE_THRESHOLD`. Either the question genuinely
isn't covered by the document, or the threshold is too tight for your content.
The printed distance scores tell you which — raise `DISTANCE_THRESHOLD` in `.env`
to just above the typical first score if good questions are being rejected.

**Answers reference the wrong document.**
You changed `PDF_PATH` without rebuilding. Run `python chat_loa.py build`.

**First run hangs with no output.**
It's downloading the embedding model from Hugging Face. Subsequent runs read it
from the local cache and start fast.

**Cohere 401 / unauthorized.**
Bad or expired key. Regenerate it in the Cohere dashboard and update `.env`.

## How it works

```
PDF ──PyPDFLoader──> pages
    ──RecursiveCharacterTextSplitter──> chunks (500 chars, 100 overlap)
    ──SentenceTransformer──> vectors
    ──faiss.IndexFlatL2──> Vector_Store/vector_db.index
                           Vector_Store/docs.csv  (chunk text + source)

question ──SentenceTransformer──> vector
         ──index.search(k=5)──> distances + chunk ids
         ──distance gate (> 1.7 → reject)
         ──chunks joined into prompt──> Cohere chat ──> answer + sources
```

`IndexFlatL2` is an exact, brute-force L2 index. It's the right choice at this
scale — no training, no approximation, no tuning. For very large corpora you'd
move to an approximate index such as `IndexIVFFlat`.
