# server.py — Amazon Reviews MCP Server (FastMCP)

import os
import re
import csv
from typing import List, Dict, Any, Optional

# --- FastMCP imports ---
try:
    from fastmcp import FastMCP
except Exception as e:
    raise SystemExit("Please install fastmcp: pip install fastmcp") from e

# Dataset location: allow env override, fallback to local file in same dir
DATA_PATH = os.getenv(
    "AMAZON_REVIEWS_CSV",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "amazon_reviews.csv")),
)

# CANGE #1: modern constructor — name only (no description kwarg)
mcp = FastMCP("amazon-reviews-mcp")

def _detect_text_column(fieldnames: List[str]) -> str:
    """
    Heuristics to find the review text column if user doesn't specify.
    Common names across public amazon review dumps.
    """
    candidates = [
        "review_body", "reviewText", "reviews.text", "text", "body",
        "review_text", "review", "content"
    ]
    lowered = {c.lower(): c for c in fieldnames}
    for c in candidates:
        if c in lowered:
            return lowered[c]
    return fieldnames[0] if fieldnames else ""

@mcp.tool()
def dataset_info() -> Dict[str, Any]:
    """Return basic information about the dataset (path, columns, row count sample)."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}. Set AMAZON_REVIEWS_CSV or place amazon_reviews.csv next to server.py"
        )
    row_count = 0
    fieldnames: List[str] = []
    with open(DATA_PATH, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for _ in reader:
            row_count += 1
            if row_count >= 50000:  # quick cap
                break
    return {
        "path": DATA_PATH,
        "approx_row_count_cap_50k": row_count,
        "columns": fieldnames,
        "default_text_column": _detect_text_column(fieldnames) if fieldnames else None,
    }

@mcp.tool()
def count_brand_mentions(brand: str, text_column: Optional[str] = None, regex: bool = False) -> Dict[str, Any]:
    """Count how many reviews mention a brand (case-insensitive)."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")
    mentions = 0
    total = 0
    with open(DATA_PATH, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        col = text_column or _detect_text_column(fieldnames)
        if not col or col not in fieldnames:
            raise ValueError(f"text_column '{col}' not found. Available: {fieldnames}")
        pattern = re.compile(brand, flags=re.IGNORECASE) if regex else re.compile(r"\b" + re.escape(brand) + r"\b", re.IGNORECASE)
        for row in reader:
            total += 1
            text = (row.get(col) or "").strip()
            if text and pattern.search(text):
                mentions += 1
    return {
        "brand": brand,
        "mentions": mentions,
        "total_reviews_scanned": total,
        "text_column_used": col,
        "pattern_type": "regex" if regex else "word-boundary",
        "ratio": (mentions / total) if total else 0.0,
    }

@mcp.tool()
def sample_reviews_with_brand(brand: str, limit: int = 5, text_column: Optional[str] = None) -> Dict[str, Any]:
    """Return up to 'limit' snippets that mention the brand (case-insensitive)."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")
    results: List[Dict[str, Any]] = []
    with open(DATA_PATH, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        col = text_column or _detect_text_column(fieldnames)
        if not col or col not in fieldnames:
            raise ValueError(f"text_column '{col}' not found. Available: {fieldnames}")
        pattern = re.compile(r"\b" + re.escape(brand) + r"\b", flags=re.IGNORECASE)
        for row in reader:
            text = (row.get(col) or "").strip()
            if text and pattern.search(text):
                results.append({"snippet": text[:400], "full_text_len": len(text)})
                if len(results) >= max(1, int(limit)):
                    break
    return {"brand": brand, "text_column_used": col, "count_returned": len(results), "examples": results}

# CHANGE #2: resource URI must be a valid URL-like string
@mcp.resource("amazon-reviews://csv")
def amazon_reviews_csv() -> str:
    """Expose the raw CSV as a text resource (first ~1MB for safety)."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")
    chunk = []
    total_bytes = 0
    with open(DATA_PATH, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            b = len(line.encode("utf-8", errors="ignore"))
            if total_bytes + b > 1_000_000:
                break
            chunk.append(line)
            total_bytes += b
    return "".join(chunk)

@mcp.tool()
def explain_result_with_llm(brand: str, provider: str = "openai", model: str = "gpt-4o-mini",
                            text_column: Optional[str] = None) -> Dict[str, Any]:
    """Use OpenAI or Ollama to generate a 1-paragraph explanation of the brand-mention finding."""
    stats = count_brand_mentions(brand=brand, text_column=text_column)
    prompt = (
        f"We analyzed Amazon user reviews to see how often the brand '{brand}' is mentioned.\n"
        f"Mentions: {stats['mentions']} out of {stats['total_reviews_scanned']} reviews (ratio={stats['ratio']:.2%}).\n"
        "Write a brief, neutral explanation of what this could suggest, including caveats about sampling bias and naming collisions.\n"
    )
    if provider == "openai":
        try:
            from openai import OpenAI
        except Exception as e:
            return {"error": "openai package not installed. pip install openai", "details": str(e), "stats": stats}
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {"error": "OPENAI_API_KEY not set", "stats": stats}
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
        message = resp.choices[0].message.content
        return {"provider": "openai", "model": model, "message": message, "stats": stats}
    elif provider == "ollama":
        import json, urllib.request
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        data = {"model": model, "prompt": prompt, "stream": False}
        req = urllib.request.Request(host + "/api/generate", data=json.dumps(data).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            out = json.loads(r.read().decode("utf-8"))
            return {"provider": "ollama", "model": model, "message": out.get("response"), "stats": stats}
    else:
        return {"error": f"Unsupported provider '{provider}'", "stats": stats}

if __name__ == "__main__":
    mcp.run()
