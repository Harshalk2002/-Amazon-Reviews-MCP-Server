# Amazon Reviews MCP Server (FastMCP)

This repo contains a **Model Context Protocol (MCP)** server built with **FastMCP** that exposes an `amazon_reviews` dataset and tools to analyze brand mentions. It also includes a small Python demo client.

## What’s included
- `server.py` — FastMCP server exposing:
  - `resource`: `amazon_reviews_csv` (first ~1MB of the CSV)
  - `tool`: `dataset_info()` — columns & quick row count
  - `tool`: `count_brand_mentions(brand, text_column=None, regex=False)` — counts mentions
  - `tool`: `sample_reviews_with_brand(brand, limit=5, text_column=None)` — example snippets
  - `tool`: `explain_result_with_llm(brand, provider='openai'|'ollama', model=...)` — optional LLM explanation
- `demo_client.py` — minimal Python client that launches the server over stdio and calls the tools
- `requirements.txt`

## Quick start

1) **Place the dataset**  
   Put your CSV at one of:
   - Same folder, named `amazon_reviews.csv`, **or**
   - Set an environment variable:
     ```bash
     export AMAZON_REVIEWS_CSV=/absolute/path/to/amazon_reviews.csv
     ```

2) **Install dependencies**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

3) **Run the server (for inspection)**
   ```bash
   python server.py
   ```
   The server speaks MCP over stdio; use an MCP-compatible client or the included demo.

4) **Run the demo client**
   ```bash
   python demo_client.py Apple
   ```
   It will:
   - list available tools
   - call `count_brand_mentions("Apple")`
   - fetch a few sample snippets
   - (optionally) ask an LLM to summarize the finding if configured

## Optional: LLM explanation
You can ask the server to generate a 1‑paragraph explanation of the metric using OpenAI or Ollama.

### OpenAI
```bash
export OPENAI_API_KEY=sk-...
python demo_client.py Apple
```
The demo calls `explain_result_with_llm(provider='openai', model='gpt-4o-mini')` via the server.

### Ollama (local)
```bash
# Ensure Ollama is running locally (defaults to http://localhost:11434)
# e.g., ollama run llama3.1
export OLLAMA_HOST=http://localhost:11434
python demo_client.py Apple
```

## Grading notes
- **FastMCP** is used to define the MCP server (`server.py`) and tools/resources.
- **Tool requirement**: `count_brand_mentions(brand)` implements the brand mention count (case‑insensitive, word‑boundary by default; can switch to regex).
- **Demo**: `demo_client.py` launches the MCP server as a subprocess via stdio and calls the tools end‑to‑end.

## Troubleshooting
- If you see `Dataset not found`, set `AMAZON_REVIEWS_CSV` or place the CSV beside `server.py`.
- If your dataset’s text column isn’t `review_body`, the server will auto‑detect a likely column
  (e.g., `reviewText`, `reviews.text`, etc.), or you can pass `text_column` explicitly.
