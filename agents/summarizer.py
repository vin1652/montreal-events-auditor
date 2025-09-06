# agents/summarizer.py
from __future__ import annotations

import os
import json
from typing import List, Optional
from datetime import datetime

import pandas as pd
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage

# ---------- Constants ----------
REPORTS_DIR = "reports"


# ---------- Model helpers ----------
def _model():
    """
    Return a Groq Chat model. If GROQ_API_KEY is missing, callers should
    handle fallback behavior (we'll return None here and skip LLM paths).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    
    model_name = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    return ChatGroq(temperature=0.2, model_name=model_name, groq_api_key=api_key)


def _load_prefs_dict(prefs_path: str) -> dict:
    try:
        with open(prefs_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ---------- Utilities ----------
def _fmt_date(val) -> str:
    """Format a date/datetime into a short English label."""
    if pd.isna(val):
        return ""
    try:
        ts = pd.to_datetime(val)
        return ts.tz_localize("America/Toronto", nonexistent="NaT", ambiguous="NaT").strftime("%a, %b %d • %H:%M")
    except Exception:
        try:
            return pd.to_datetime(val).strftime("%a, %b %d • %H:%M")
        except Exception:
            return str(val)


def _rows_to_min_json(df: pd.DataFrame) -> List[dict]:
    """
    Converts rows to a compact JSON structure the LLM can reason over.
    Keep only useful fields; translate is handled by the LLM in the prompts.
    """
    out = []
    for _, r in df.iterrows():
        out.append({
            "title": (r.get("titre") or "")[:200],
            "url": (r.get("url_fiche") or ""),
            "borough": (r.get("arrondissement") or ""),
            "event_type": (r.get("type_evenement") or ""),
            "start": str(r.get("date_debut") or ""),
            "is_free": bool(r.get("is_free", False)),
            "audience": (r.get("public_cible") or ""),
            "temp_c": None if pd.isna(r.get("temp_c")) else float(r.get("temp_c")),
            "rain_prob": None if pd.isna(r.get("rain_prob")) else float(r.get("rain_prob")),
            "desc": (r.get("description") or "")[:600]
        })
    return out


# ---------- LLM selection (deciding which events to keep) ----------
def select_events_with_llm(
    df_short: pd.DataFrame,
    prefs_path: str,
    final_n: int = 5
) -> Optional[List[str]]:
    """
    Ask the LLM to pick the best `final_n` events from the shortlist.
    Returns a list of selected URLs (strings) or None on failure.
    """
    if df_short.empty:
        return []

    llm = _model()
    if llm is None:
        # No API key → caller should fallback
        return None

    prefs = _load_prefs_dict(prefs_path)
    likes = (prefs.get("likes") or "").strip()
    hard = prefs.get("hard_filters", {})
    borough_order = hard.get("arrondissement_allow", [])

    system = SystemMessage(content=(
        "You are an event selector and outing planner. You choose a final set of events in Montreal that closely fit the user's preferences.\n"
        "Output must be **JSON only**, no extra prose. English only."
    ))
    human = HumanMessage(content=(
        "Task: From the shortlist, pick the best events for the user.\n"
        f"- Return ONLY JSON: {{\"selected_urls\": [\"<url1>\", \"<url2>\", ...]}}\n"
        f"- Choose exactly {final_n} items if possible; if shortlist is smaller, choose all.\n"
        "- Consider: user likes (free-text), borough preference order (earlier is better), weather (avoid heavy rain for outdoor items),\n"
        "  audience, event types, price (prefer free/low-cost when appropriate), and variety across picks if possible.\n"
        "- Translate French internally if needed, but output JSON only.\n\n"
        "User likes (free text):\n"
        f"{likes}\n\n"
        "Borough preference order (earlier is better):\n"
        f"{json.dumps(borough_order, ensure_ascii=False)}\n\n"
        "Shortlist (array of events as JSON):\n"
        f"{json.dumps(_rows_to_min_json(df_short), ensure_ascii=False)}\n\n"
        "Respond with JSON ONLY like:\n"
        "{\"selected_urls\": [\"https://example.com/event1\", \"https://example.com/event2\"]}"
    ))

    try:
        resp = llm.invoke([system, human])
        text = (resp.content or "").strip()
        data = json.loads(text)
        urls = data.get("selected_urls", [])
        # Validate against shortlist
        keep_set = set(df_short["url"].fillna("").astype(str).tolist() + df_short["url_fiche"].fillna("").astype(str).tolist() if "url_fiche" in df_short.columns else [])
        urls = [u for u in urls if u in keep_set]
        return urls
    except Exception as e:
        print("LLM selection parse failed:", repr(e))
        return None


# ---------- TL;DR newsletter Generation Function (English-only) ----------
def _compose_prompt(bullets: List[str]) -> List:
    """
    Compose a prompt for English-only Markdown newsletter generation.
    """
    system = SystemMessage(content=(
        "You are a concise newsletter editor.\n"
        "Write in clear, accessible **English only** (no French or bilingual output).\n"
        "Translate any French titles/descriptions to natural English, preserving proper nouns and venue names.\n"
        "Style: scannable Markdown, short lines, neutral/helpful tone, no hype, no invented facts."
    ))
    human = HumanMessage(content=(
        "Turn the following event bullets into a short weekly TL;DR with the sections:\n"
        "1) Top Picks  2) Free or Low-Cost  3) Outdoor Picks (note temp/rain)\n\n"
        "Rules:\n"
        "- Output valid **English** Markdown only (no YAML front matter).\n"
        "- Intro: 1–2 sentences max.\n"
        "- Under each section, list 3–5 bullets chosen from the input; avoid duplicates.\n"
        "- Do not add events that are not present in the input.\n"
        "- Keep bullets to one line each; include borough and weather notes if provided.\n\n"
        "EVENT BULLETS:\n" + "\n".join(bullets)
    ))
    return [system, human]


def _default_intro(run_iso: str) -> str:
    date_label = run_iso[:10] if run_iso else datetime.now().strftime("%Y-%m-%d")
    return f"# Montréal Events — Week of {date_label}\n\nHere are top picks tailored to your preferences. Weather notes are approximate.\n"


def _build_event_bullets(df: pd.DataFrame, limit: int = 20) -> List[str]:
    """
    Create compact, English-ready bullets from df rows.
    (The LLM will still translate any remaining FR words per prompt.)
    """
    needed = ["titre", "arrondissement", "type_evenement", "url_fiche", "venue_full", "is_free", "start_datetime", "temp_c", "rain_prob"]
    for c in needed:
        if c not in df.columns:
            df[c] = None

    bullets = []
    for _, r in df.head(limit).iterrows():
        title = ( r.get("titre") or "").strip()
        url = ( r.get("url_fiche") or "").strip()
        boro = ( r.get("arrondissement") or "").strip()
        etype = ( r.get("type_evenement") or "").strip()
        start = _fmt_date(r["start_datetime"] or r.get("date_debut"))
        venue = (r["venue_full"] or "").strip()
        tags = []
        if bool(r.get("is_free", False)):
            tags.append("free")
        if pd.notna(r.get("temp_c")):
            try:
                tags.append(f"{float(r['temp_c']):.1f}°C")
            except Exception:
                pass
        if pd.notna(r.get("rain_prob")):
            try:
                tags.append(f"{int(r['rain_prob'])}% rain")
            except Exception:
                pass
        tag_str = (" — " + ", ".join(tags)) if tags else ""
        loc_str = f" ({boro})" if boro else ""
        venue_str = f" — {venue}" if venue else ""
        etype_str = f" — {etype}" if etype else ""

        bullet = f"- **{title}**{loc_str} — {start}{etype_str}{venue_str}{tag_str}\n  {url}"
        bullets.append(bullet)

    return bullets


def summarize_to_markdown(df: pd.DataFrame, run_iso: str) -> str:
    """
    Produce the final English Markdown TL;DR for the selected events in `df`.
    Uses Groq LLM if available; otherwise falls back to a simple list.
    """
    if df is None or df.empty:
        return _default_intro(run_iso) + "\n_No events matched your filters this week._\n"

    bullets = _build_event_bullets(df, limit=max(10, len(df)))

    llm = _model()
    if llm is None:
        # Fallback: simple English list without LLM
        intro = _default_intro(run_iso)
        body = "## Top Picks\n" + "\n".join(bullets[: min(5, len(bullets))])
        return intro + "\n" + body + "\n"

    try:
        messages = _compose_prompt(bullets)
        resp = llm.invoke(messages)
        text = resp.content or ""
        if not text.strip():
            raise ValueError("Empty LLM response")
        return text
    except Exception as e:
        print("Summarization failed, using fallback:", repr(e))
        intro = _default_intro(run_iso)
        body = "## Top Picks\n" + "\n".join(bullets[: min(5, len(bullets))])
        return intro + "\n" + body + "\n"


# ---------- Save report ----------
def save_report(md: str, run_iso: str) -> str:
    """
    Save the Markdown to a fixed filename so the workflow overwrites weekly.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, "weekly_tldr.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return path
