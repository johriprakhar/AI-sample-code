# Screenshots

Reference images for the web UI (`app.py`). Committed so the main README can
show the interface without anyone needing to run the app.

| File | What it shows |
| --- | --- |
| `web-ui-answer.png` | The search page with an answer rendered: status banner, question box, answer, sources, distance metadata, retrieved-context panel. |
| `web-ui-loader-stuck-bug.png` | The 2026-10-09 bug report — spinner still reading "Searching the document…" after the answer had arrived, plus an empty red error banner above the result. Kept as the before image for that fix. |

## How to capture a replacement

1. `python chat_loa.py build` (once), then `python app.py`.
2. Open <http://127.0.0.1:5000> and ask a question that returns an answer.
3. Capture the full page, not just the viewport — in Chrome or Edge DevTools:
   <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd> → "Capture full size screenshot".
4. Save as PNG into this folder using the filename above, so the README link
   keeps working.

Keep images under about 500 KB. These are UI references, not print assets.
