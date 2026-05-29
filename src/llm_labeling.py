"""
llm_labeling.py - Stage 3: Weak supervision via Gemini Flash.

Design decisions:
  - Model: gemini-1.5-flash  (free tier, fast, sufficient for classification)
  - Input: summary_short (concise, fits well in prompt)
  - Output: one of 6 fixed category labels
  - Prompt style: few-shot with category definitions + examples
  - Batch: 1 repo per call to avoid token limits and improve reliability
  - Rate limiting: 1 req/sec to stay in free tier

Prompt engineering rationale:
  The prompt gives the LLM:
    1. A clear task description
    2. Explicit category definitions with distinguishing signals
    3. Few-shot examples (one per category)
    4. The repo summary
    5. Strict output format instruction (just the label, nothing else)

  This avoids hallucination and makes parsing trivial.

Limitations (required analysis):
  - LLM may be biased toward "senior" for repos with many stars
  - LLM cannot see the actual code, only metadata signals
  - Some repos fall on category boundaries — LLM will pick one arbitrarily
  - Gemini free tier has 15 RPM limit; we sleep accordingly
"""

import time
import re
import pandas as pd
import google.generativeai as genai
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent))
from utils import get_logger, get_gemini_key, PATHS, CATEGORIES, CATEGORY_DESCRIPTIONS

logger = get_logger("llm_labeling")

VALID_LABELS = set(CATEGORIES)

# ── Prompt construction ───────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """
Examples:

Repository summary: "Repo 'alice/hello-world' | Python | 3 months old | 0 stars, 0 forks, 1 contributor | 5 commits/year | CI: no | Releases: 0 | README: yes (size 200B) | Topics: no topics | Description: my first python project"
Label: intern

Repository summary: "Repo 'bob/todo-app' | JavaScript | 8 months old | 12 stars, 3 forks, 2 contributors | 45 commits/year | CI: no | Releases: 1 | README: yes (size 1800B) | Topics: react, todo | Description: a simple todo application built with React"
Label: junior

Repository summary: "Repo 'carol/fastapi-microservice' | Python | 2y 1m old | 280 stars, 42 forks, 8 contributors | 320 commits/year | CI: yes | Releases: 15 | README: yes (size 8500B) | Topics: fastapi, microservices, docker | Description: production-ready FastAPI microservice template with auth, tests and CI"
Label: senior

Repository summary: "Repo 'dave-org/distributed-systems-framework' | Go | 4y 3m old | 2100 stars, 380 forks, 47 contributors | 890 commits/year | CI: yes | Releases: 62 | README: yes (size 22000B) | Topics: distributed-systems, consensus, raft | Description: battle-tested distributed consensus framework used in production"
Label: lead

Repository summary: "Repo 'eve/react-starter' | JavaScript | 1y 0m old | 5 stars, 30 forks, 1 contributor | 2 commits/year | CI: no | Releases: 0 | README: yes (size 1200B) | Topics: template, starter, react | Description: fork of create-react-app with minor changes"
Label: template_or_clone

Repository summary: "Repo 'frank/test123' | unknown age | 0 stars, 0 forks, 0 contributors | 0 commits/year | CI: no | Releases: 0 | README: no (size 0B) | Topics: no topics | Description: no description"
Label: low_value
"""

SYSTEM_PROMPT = f"""You are an expert engineering talent analyst. Your task is to classify GitHub repositories by their engineering maturity level.

Categories and their definitions:
- intern: Beginner-level work. Simple scripts, no project structure, no tests, minimal or no documentation. Usually a single contributor with very few commits.
- junior: Basic project structure is present. Some documentation, minimal tests, possibly one or two contributors. Limited CI/CD. Early-stage professional work.
- senior: Well-structured project. Clear documentation, CI/CD pipelines, tests, multiple contributors, regular releases. Shows professional engineering practices.
- lead: Complex architecture. Many contributors, high commit frequency, many releases, extensive documentation, demonstrates system design and long-term maintenance. Could be open source projects used in production.
- template_or_clone: A forked repo with no meaningful changes, a boilerplate starter, or a repo that clearly replicates another project.
- low_value: Empty, abandoned, trivial, or otherwise not worth reviewing. Very few or zero commits, stars, contributors, and no meaningful content.

{FEW_SHOT_EXAMPLES}

Instructions:
- Read the repository summary carefully.
- Respond with ONLY one label from this list: intern, junior, senior, lead, template_or_clone, low_value
- Do not include any explanation, punctuation, or extra text. Just the label.
"""


def build_user_message(summary: str) -> str:
    return f"Repository summary: \"{summary}\"\nLabel:"


# ── Gemini client ─────────────────────────────────────────────────────────────

def init_gemini():
    key = get_gemini_key()
    genai.configure(api_key=key)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,     # low temp for consistent classification
            max_output_tokens=20,
        ),
    )
    return model


def classify_repo(model, summary: str, retries: int = 3) -> str:
    """Call Gemini and return a validated label."""
    user_msg = build_user_message(summary)
    for attempt in range(retries):
        try:
            response = model.generate_content(user_msg)
            label = response.text.strip().lower()
            # Clean punctuation
            label = re.sub(r"[^a-z_]", "", label)
            if label in VALID_LABELS:
                return label
            # Try to find a valid label inside the response
            for candidate in VALID_LABELS:
                if candidate in label:
                    return candidate
            logger.warning(f"Unexpected label: {label!r} — defaulting to 'low_value'")
            return "low_value"
        except Exception as e:
            logger.warning(f"Gemini error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return "low_value"


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(limit: int = None):
    """
    Label all repositories using Gemini.

    Args:
        limit: if set, only label this many repos (useful for testing)
    """
    summ_path = PATHS["processed"] / "repositories_summarized.csv"
    if not summ_path.exists():
        raise FileNotFoundError(f"Summarized data not found at {summ_path}. Run summarization.py first.")

    df = pd.read_csv(summ_path, low_memory=False)
    if limit:
        df = df.head(limit)

    logger.info(f"Labeling {len(df)} repositories with Gemini …")
    model = init_gemini()

    labels = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="LLM labeling"):
        label = classify_repo(model, str(row["summary_short"]))
        labels.append(label)
        time.sleep(1.1)   # ~55 req/min, well within free tier 60 RPM

    df["label"] = labels

    # Distribution report
    dist = df["label"].value_counts()
    logger.info(f"Label distribution:\n{dist.to_string()}")

    out_path = PATHS["labeled"] / "repositories_labeled.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved labeled data to {out_path}")
    return df


if __name__ == "__main__":
    # Pass limit=50 for a quick test
    df = run()
    print(df["label"].value_counts())
