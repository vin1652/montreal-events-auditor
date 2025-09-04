# agents/cleaner.py
from __future__ import annotations
import pandas as pd
import numpy as np

def clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal adapter:
      - KEEP all original French columns unchanged.
      - ADD a few lightweight alias columns the pipeline expects:
          title, url, description, start_datetime, end_datetime,
          event_type, audience, borough, venue, venue_full,
          cost, is_free, lat, lon
      - Parse only the alias datetime columns so time filters work.
      - Do NOT dedupe, do NOT trim strings, do NOT drop rows.
    """
    df = df_raw.copy()

    # ---- Aliases (new columns) ----
    # Text / categorical
    df["title"]        = df.get("titre")
    df["url"]          = df.get("url_fiche")
    df["description"]  = df.get("description")
    df["event_type"]   = df.get("type_evenement")
    df["audience"]     = df.get("public_cible")
    df["borough"]      = df.get("arrondissement")
    df["venue"]        = df.get("titre_adresse")
    df["cost"]         = df.get("cout")

    # Location / coords
    df["lat"]          = pd.to_numeric(df.get("lat"), errors="coerce")
    # Original is "long" in the dataset — create a "lon" alias some libs expect
    if "long" in df.columns:
        df["lon"] = pd.to_numeric(df["long"], errors="coerce")
    else:
        df["lon"] = np.nan

    # Datetimes (aliases only; originals remain untouched)
    df["start_datetime"] = pd.to_datetime(df.get("date_debut"), errors="coerce")
    df["end_datetime"]   = pd.to_datetime(df.get("date_fin"),   errors="coerce")

    # Simple helper flags used by the newsletter (optional, cheap)
    cost_lower = df["cost"].astype(str).str.lower()
    df["is_free"] = cost_lower.str.contains("gratuit", na=False)

    # One handy display line for venue (keeps original fields intact)
    df["venue_full"] = [
        ", ".join([str(v) for v in [row.get("titre_adresse"), row.get("arrondissement")] if pd.notna(v) and str(v)])
        for _, row in df.iterrows()
    ]

    return df
