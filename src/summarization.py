"""
summarization.py - Stage 2b: Generate text representations of each repository.

Why text summaries?
  Both the LLM labeler and the BERT classifier need text input.
  Rather than dumping raw JSON, we craft a structured natural-language
  description that highlights the signals most relevant to engineering maturity.
  This makes the LLM prompt more reliable and gives BERT meaningful tokens.

Two representations are produced:
  1. `summary_short`  - 1-2 sentence description (used in LLM prompt)
  2. `summary_full`   - detailed paragraph (used as BERT input)
"""

import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent))
from utils import get_logger, PATHS

logger = get_logger("summarization")


def _bool_phrase(val: bool, true_str: str, false_str: str) -> str:
    return true_str if val else false_str


def _age_phrase(days: float) -> str:
    if pd.isna(days):
        return "unknown age"
    years = int(days // 365)
    months = int((days % 365) // 30)
    if years > 0:
        return f"{years}y {months}m old"
    return f"{months} months old"


def _activity_phrase(days_since_push: float) -> str:
    if pd.isna(days_since_push):
        return "unknown activity"
    if days_since_push < 30:
        return "actively maintained (pushed within last month)"
    if days_since_push < 180:
        return "recently active (pushed within 6 months)"
    if days_since_push < 365:
        return "moderately active (pushed within a year)"
    return f"inactive (last push {int(days_since_push)} days ago)"


def build_summary_short(row: pd.Series) -> str:
    """One-line summary used inside the LLM prompt."""
    topics_str = ", ".join(row["topics"][:5]) if row["topics"] else "no topics"
    lang = row["language"] or "unspecified language"
    desc = row["description"].strip()[:120] if row["description"] else "no description"

    return (
        f"Repo '{row['full_name']}' | {lang} | {_age_phrase(row['repo_age_days'])} | "
        f"{row['stargazers_count']} stars, {row['forks_count']} forks, "
        f"{row['contributors_count']} contributors | "
        f"{row['commit_count_last_year']} commits/year | "
        f"CI: {'yes' if row['has_ci_workflow'] else 'no'} | "
        f"Releases: {row['release_count']} | "
        f"README: {'yes' if row['has_readme'] else 'no'} (size {row['readme_size_bytes']}B) | "
        f"Topics: {topics_str} | "
        f"Description: {desc}"
    )


def build_summary_full(row: pd.Series) -> str:
    """
    Full paragraph used as BERT classifier input.
    Designed to be information-dense but readable.
    """
    topics_str = ", ".join(row["topics"][:8]) if row["topics"] else "none"
    lang = row["language"] or "unspecified"
    desc = row["description"].strip() if row["description"] else "No description provided."
    license_str = row["license"] if row["license"] else "no license"

    fork_note = "This is a forked repository. " if row["is_fork"] else ""
    archived_note = "The repository is archived (read-only). " if row["is_archived"] else ""
    template_note = "It appears to be a template or boilerplate. " if row["is_likely_template"] else ""
    abandoned_note = "It appears abandoned with no recent activity. " if row["is_abandoned"] else ""

    return (
        f"{fork_note}{archived_note}{template_note}{abandoned_note}"
        f"Repository: {row['full_name']}. "
        f"Description: {desc} "
        f"Primary language: {lang}. "
        f"Age: {_age_phrase(row['repo_age_days'])}. "
        f"Activity: {_activity_phrase(row['days_since_last_push'])}. "
        f"Community: {row['stargazers_count']} stars, {row['forks_count']} forks, "
        f"{row['watchers_count']} watchers. "
        f"Team: {row['contributors_count']} contributors. "
        f"Development: {row['commit_count_last_year']} commits in the last year, "
        f"{row['open_prs_count']} open pull requests, "
        f"{row['release_count']} releases. "
        f"Quality signals: "
        f"{'has CI/CD workflows' if row['has_ci_workflow'] else 'no CI/CD workflows'}, "
        f"{'has README' if row['has_readme'] else 'no README'} "
        f"({'detailed' if row['readme_size_bytes'] > 3000 else 'minimal'} documentation), "
        f"{'has wiki' if row['has_wiki'] else 'no wiki'}, "
        f"license: {license_str}. "
        f"Maturity score: {row['maturity_score']}/5. "
        f"Documentation score: {row['doc_score']}/3. "
        f"Repository size: {row['size_kb']} KB. "
        f"Topics: {topics_str}. "
        f"Open issues: {row['open_issues_count']}."
    )


def run():
    proc_path = PATHS["processed"] / "repositories_processed.csv"
    if not proc_path.exists():
        raise FileNotFoundError(f"Processed data not found at {proc_path}. Run preprocessing.py first.")

    df = pd.read_csv(proc_path, low_memory=False)

    # Parse list columns that were stringified by CSV
    import ast
    df["topics"] = df["topics"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
    )

    logger.info(f"Building summaries for {len(df)} repositories …")
    df["summary_short"] = df.apply(build_summary_short, axis=1)
    df["summary_full"]  = df.apply(build_summary_full, axis=1)

    out_path = PATHS["processed"] / "repositories_summarized.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved summarized data to {out_path}")
    return df


if __name__ == "__main__":
    df = run()
    # Print a sample
    sample = df.sample(3)
    for _, row in sample.iterrows():
        print("=" * 80)
        print(row["summary_full"])
        print()
