"""
preprocessing.py - Stage 2: Clean raw data and engineer features.

This module:
  1. Loads raw repository data
  2. Handles missing values and type coercion
  3. Engineers derived features (repo age, activity ratios, etc.)
  4. Saves the processed dataset ready for summarization and labeling
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

import sys
sys.path.append(str(Path(__file__).parent))
from utils import get_logger, PATHS

logger = get_logger("preprocessing")

REFERENCE_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _parse_date(val) -> datetime | None:
    if pd.isna(val) or not val:
        return None
    try:
        dt = pd.to_datetime(val, utc=True)
        return dt.to_pydatetime()
    except Exception:
        return None


def _days_since(dt: datetime | None) -> float:
    if dt is None:
        return np.nan
    return (REFERENCE_DATE - dt).days


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: types, nulls, deduplication."""
    logger.info(f"Input shape: {df.shape}")

    # Drop exact duplicates
    df = df.drop_duplicates(subset="full_name").reset_index(drop=True)

    # Numeric coercion
    numeric_cols = [
        "stargazers_count", "forks_count", "open_issues_count",
        "watchers_count", "size_kb", "contributors_count",
        "commit_count_last_year", "readme_size_bytes",
        "open_prs_count", "release_count",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Boolean coercion
    bool_cols = [
        "is_fork", "is_archived", "has_wiki", "has_projects",
        "has_discussions", "has_readme", "has_ci_workflow",
    ]
    for col in bool_cols:
        df[col] = df[col].astype(bool)

    # String cols
    for col in ["description", "language", "license", "default_branch"]:
        df[col] = df[col].fillna("").astype(str)

    # Topics: ensure list
    def _parse_topics(t):
        if isinstance(t, list):
            return t
        if isinstance(t, str):
            try:
                import ast
                return ast.literal_eval(t)
            except Exception:
                return []
        return []

    df["topics"] = df["topics"].apply(_parse_topics)

    logger.info(f"After dedup & cleaning: {df.shape}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive higher-level signals from raw ones."""

    # Date-based features
    df["created_at_dt"] = df["created_at"].apply(_parse_date)
    df["pushed_at_dt"]  = df["pushed_at"].apply(_parse_date)
    df["updated_at_dt"] = df["updated_at"].apply(_parse_date)

    df["repo_age_days"]          = df["created_at_dt"].apply(_days_since)
    df["days_since_last_push"]   = df["pushed_at_dt"].apply(_days_since)
    df["days_since_last_update"] = df["updated_at_dt"].apply(_days_since)

    # Drop datetime objects (not needed downstream)
    df.drop(columns=["created_at_dt", "pushed_at_dt", "updated_at_dt"], inplace=True)

    # Activity ratio: commits per month of life
    df["commits_per_month"] = df.apply(
        lambda r: r["commit_count_last_year"] / max(r["repo_age_days"] / 30, 1),
        axis=1,
    ).round(2)

    # Documentation score (0-3)
    df["doc_score"] = (
        df["has_readme"].astype(int)
        + (df["readme_size_bytes"] > 2000).astype(int)   # non-trivial README
        + df["has_wiki"].astype(int)
    )

    # Engineering maturity score (0-5)
    df["maturity_score"] = (
        df["has_ci_workflow"].astype(int)
        + (df["release_count"] > 0).astype(int)
        + (df["contributors_count"] > 1).astype(int)
        + (df["open_prs_count"] > 0).astype(int)
        + df["has_projects"].astype(int)
    )

    # Community score (log-scaled stars + forks)
    df["community_score"] = (
        np.log1p(df["stargazers_count"]) + np.log1p(df["forks_count"])
    ).round(2)

    # Topics count
    df["topics_count"] = df["topics"].apply(len)

    # Is it likely a template or clone?
    df["is_likely_template"] = (
        df["is_fork"]
        | df["topics"].apply(lambda t: "template" in t or "boilerplate" in t or "starter" in t)
    )

    # Is it abandoned?
    df["is_abandoned"] = (
        (df["days_since_last_push"] > 365)
        & (df["stargazers_count"] < 5)
        & (df["contributors_count"] <= 1)
    )

    logger.info(f"Feature engineering complete. Final shape: {df.shape}")
    return df


def run():
    raw_path = PATHS["raw"] / "repositories_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}. Run github_collector.py first.")

    df = pd.read_csv(raw_path, low_memory=False)
    df = clean(df)
    df = engineer_features(df)

    out_path = PATHS["processed"] / "repositories_processed.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved processed data to {out_path}")
    return df


if __name__ == "__main__":
    df = run()
    print(df.dtypes)
    print(df.describe())
