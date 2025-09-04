# agents/summarizer.py
from __future__ import annotations
import os
from typing import List
import pandas as pd
from dotenv import load_dotenv
from jinja2 import Template
from datetime import datetime, timezone

# LLM: Groq (LangChain wrapper)
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

# ------------------------------
# Config
# ------------------------------
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")  # fast + good-enough
MAX_EVENTS_FOR_SUMMARY = int(os.getenv("MAX_EVENTS_FOR_SUMMARY", "12"))  # keep prompt small
REPORTS_DIR = "reports"


# ------------------------------
# Helpers
# ------------------------------
def _fmt_date(d) -> str:
    try:
        d = pd.to_datetime(d)
        return d.strftime("%a %b %d, %H:%M")
    except Exception:
        return str(d)

def _build_event_bullets(df: pd.DataFrame, limit: int = MAX_EVENTS_FOR_SUMMARY) -> List[str]:
    """
    Turn ranked rows into compact bullets to keep token usage small.
    We only keep what's most helpful for a weekly TL;DR.
    """
    cols_needed = ["title", "start_datetime", "borough", "event_type", "url", "venue_full", "is_free", "temp_c", "rain_prob"]
    for c in cols_needed:
        if c not in df.columns:
            df[c] = None

    bullets = []
    for _, r in df.head(limit).iterrows():
        title = (r["title"] or "").strip()
        start = _fmt_date(r["start_datetime"])
        boro = r["borough"] or ""
        etype = r["event_type"] or ""
        venue = r["venue_full"] or ""
        url = r["url"] or ""
        tags = []
        if r["is_free"]:
            tags.append("free")
        if pd.notna(r.get("temp_c")):
            tags.append(f"{float(r['temp_c']):.1f}°C")
        if pd.notna(r.get("rain_prob")):
            tags.append(f"{int(r['rain_prob'])}% rain")

        tag_str = (" — " + ", ".join(tags)) if tags else ""
        loc_str = f" ({boro})" if boro else ""
        venue_str = f" — {venue}" if venue else ""
        etype_str = f" — {etype}" if etype else ""

        bullet = f"- **{title}**{loc_str} — {start}{etype_str}{venue_str}{tag_str}\n  {url}"
        bullets.append(bullet)
    return bullets

def _default_intro(run_iso: str) -> str:
    date_label = run_iso[:10]
    return f"# Montréal Events — Week of {date_label}\n\nHere are top picks tailored to your preferences. Weather notes are approximate.\n"

def _render_markdown_without_llm(bullets: List[str], run_iso: str) -> str:
    """Fallback rendering if LLM key is missing; still produces a usable report."""
    intro = _default_intro(run_iso)
    body = "\n".join(bullets)
    return intro + body + "\n"

def _make_llm() -> ChatGroq:
    # load local .env (no-op on GitHub Actions)
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found. Locally, create a .env with GROQ_API_KEY=... ; on GitHub, add it as an Actions secret.")
    return ChatGroq(groq_api_key=api_key, model=DEFAULT_MODEL, temperature=0.2, timeout=45)

def _compose_prompt(bullets: List[str]) -> List:
    system = SystemMessage(content=(
        "You are a concise newsletter editor.\n"
        "Write in clear, accessible **English only** (no French or bilingual output).\n"
        "If event titles/descriptions are in French, translate them to natural English, preserving proper nouns.\n"
        "Style: scannable Markdown, short lines, neutral/helpful tone, no hype, no invented facts."
    ))
    human = HumanMessage(content=(
        "Turn the following event bullets into a short weekly TL;DR newsletter with sections:\n"
        "1) Top 5 Picks  2) Free or Low-Cost  3) Outdoor Picks (note temp/rain)\n\n"
        "Rules:\n"
        "- Output valid **English** Markdown only (no YAML front matter).\n"
        "- Keep the intro to 1–2 sentences in English.\n"
        "- Under each section, list 3–5 bullets chosen from the input; avoid duplicates.\n"
        "- Do not add events not present in the input.\n"
        "- Translate French content to English, but keep proper names (venues/areas) as-is.\n"
        "- Keep bullets to one line each; include borough and weather notes if provided.\n\n"
        "EVENT BULLETS:\n" + "\n".join(bullets)
    ))
    return [system, human]

def summarize_to_markdown(df_ranked: pd.DataFrame, run_iso: str) -> str:
    """
    Main entrypoint. Returns Markdown string.
    If GROQ_API_KEY is missing, falls back to deterministic, non-LLM rendering.
    """
    bullets = _build_event_bullets(df_ranked, limit=MAX_EVENTS_FOR_SUMMARY)
    if not bullets:
        return _default_intro(run_iso) + "_No events found this week._\n"

    try:
        llm = _make_llm()
        messages = _compose_prompt(bullets)
        resp = llm.invoke(messages)
        md = resp.content or ""
        # Light sanity check: ensure it’s markdown-ish; if not, fallback
        if not md.strip().startswith("#"):
            md = _default_intro(run_iso) + md
        return md
    except Exception:
        # If key missing or network issues, degrade gracefully
        return _render_markdown_without_llm(bullets, run_iso)

def save_report(md: str, run_iso: str) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    # Use a fixed filename:
    path = os.path.join(REPORTS_DIR, "weekly_tldr.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return path
