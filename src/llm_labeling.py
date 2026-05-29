"""
llm_labeling.py - Stage 3: Weak supervision via Gemini.
"""

import time
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from google import genai
from google.genai import types

import sys
sys.path.append(str(Path(__file__).parent))
from utils import get_logger, get_gemini_key, PATHS, CATEGORIES

logger = get_logger("llm_labeling")

VALID_LABELS = set(CATEGORIES)

SYSTEM_PROMPT = """You are an expert engineering talent analyst. Classify GitHub repositories by engineering maturity.

Categories:
- intern: Simple scripts, no structure, no tests, minimal docs, single contributor, very few commits.
- junior: Basic project structure, some docs, minimal tests, few contributors, limited CI/CD.
- senior: Well structured, CI/CD, tests, clear docs, multiple contributors, regular releases.
- lead: Complex architecture, many contributors, many releases, extensive docs, widely used in production.
- template_or_clone: Forked with no meaningful changes, or a boilerplate/starter kit.
- low_value: Empty, abandoned, or trivial repository.

Examples:
Summary: "0 stars, 0 forks, 1 contributor, 5 commits/year, CI: no, Releases: 0, README: yes (200B)"
Label: intern

Summary: "12 stars, 3 forks, 2 contributors, 45 commits/year, CI: no, Releases: 1, README: yes (1800B)"
Label: junior

Summary: "280 stars, 42 forks, 8 contributors, 320 commits/year, CI: yes, Releases: 15, README: yes (8500B)"
Label: senior

Summary: "2100 stars, 380 forks, 47 contributors, 890 commits/year, CI: yes, Releases: 62, README: yes (22000B)"
Label: lead

Summary: "5 stars, 30 forks, 1 contributor, 2 commits/year, CI: no, Releases: 0, topic: template"
Label: template_or_clone

Summary: "0 stars, 0 forks, 0 contributors, 0 commits/year, CI: no, Releases: 0, README: no"
Label: low_value

Respond with ONLY one label. No explanation, no punctuation. Just the label."""


def init_client():
    key = get_gemini_key()
    return genai.Client(api_key=key)


def classify_repo(client, summary: str, retries: int = 3) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\nSummary: \"{summary}\"\nLabel:"
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=10,
                ),
            )
            label = response.text.strip().lower()
            label = re.sub(r"[^a-z_]", "", label)
            if label in VALID_LABELS:
                return label
            for candidate in VALID_LABELS:
                if candidate in label:
                    return candidate
            return "low_value"
        except Exception as e:
            logger.warning(f"Gemini error (attempt {attempt+1}): {e}")
            time.sleep(4 ** attempt)
    return "low_value"


def run(limit: int = None):
    summ_path = PATHS["processed"] / "repositories_summarized.csv"
    if not summ_path.exists():
        raise FileNotFoundError(f"Not found: {summ_path}")

    df = pd.read_csv(summ_path, low_memory=False)
    if limit:
        df = df.head(limit)

    logger.info(f"Labeling {len(df)} repositories with Gemini ...")
    client = init_client()

    labels = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="LLM labeling"):
        label = classify_repo(client, str(row["summary_short"]))
        labels.append(label)
        time.sleep(2)  # 30 req/min, seguro para tier gratuito

    df["label"] = labels
    dist = df["label"].value_counts()
    logger.info(f"Label distribution:\n{dist.to_string()}")

    out_path = PATHS["labeled"] / "repositories_labeled.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved labeled data to {out_path}")
    return df


if __name__ == "__main__":
    df = run()
    print(df["label"].value_counts())