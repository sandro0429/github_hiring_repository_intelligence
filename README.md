# GitHub Hiring Repository Intelligence

**Track A — Engineering Maturity Classification using Weak Supervision + DistilBERT**

A complete NLP pipeline that analyzes GitHub repositories and classifies them by engineering maturity level to assist recruiters, engineering managers, startups, and accelerators.

---

## What does this project do?

This system evaluates GitHub repositories and classifies each one into one of six engineering maturity categories:

| Category | Description |
|---|---|
| `intern` | Simple scripts, no structure, no tests, single contributor |
| `junior` | Basic project structure, some docs, minimal tests |
| `senior` | CI/CD, tests, clear docs, multiple contributors, regular releases |
| `lead` | Complex architecture, many contributors, many releases, widely used |
| `template_or_clone` | Forked or boilerplate with no meaningful changes |
| `low_value` | Empty, abandoned, or trivial repository |

The goal is **not** to judge the developer — it is to estimate the engineering maturity reflected by the repository itself.

---

## Track Selected

**Track A: Hiring-Oriented Repository Intelligence**

---

## Repositories Analyzed

Repositories were collected using the GitHub REST Search API across 10 search queries designed to sample all maturity levels:

- High-quality: `stars:>500 pushed:>2024-01-01 language:Python/TypeScript`
- ML/DevOps focused: `topic:machine-learning`, `topic:devops`
- Junior/Intern level: `stars:1..20 language:Python/JavaScript`
- Templates: `topic:template stars:<50`
- Abandoned: `stars:0 pushed:<2022-01-01 size:<100`

Target: ~600 unique repositories.

---

## GitHub Signals Used

23 signals are collected per repository:

| Signal | Why it matters |
|---|---|
| `stargazers_count` | Community adoption |
| `forks_count` | Reuse and visibility |
| `contributors_count` | Team size and organizational complexity |
| `commit_count_last_year` | Development velocity |
| `has_ci_workflow` | Professional engineering practices |
| `release_count` | Product maturity and versioning discipline |
| `readme_size_bytes` | Documentation investment |
| `open_prs_count` | Collaborative development practices |
| `repo_age_days` | Project longevity |
| `days_since_last_push` | Activity / abandonment signal |
| `is_fork` | Clone detection |
| `is_archived` | Abandoned project signal |
| `has_wiki` | Extended documentation |
| `has_readme` | Basic documentation presence |
| `open_issues_count` | Community engagement |
| `size_kb` | Project scope |
| `topics` | Domain and ecosystem context |
| `language` | Tech stack |
| `license` | Open source maturity |
| `maturity_score` (derived) | Composite: CI + releases + contributors + PRs |
| `doc_score` (derived) | Composite: README + wiki + README size |
| `community_score` (derived) | log(stars) + log(forks) |
| `commits_per_month` (derived) | Activity rate normalized by age |

---

## How Summaries Were Created

Each repository is converted into two text representations:

**`summary_short`** — Used for LLM labeling. One structured line:
```
Repo 'user/project' | Python | 2y 1m old | 280 stars, 42 forks, 8 contributors |
320 commits/year | CI: yes | Releases: 15 | README: yes (size 8500B) | Topics: fastapi, docker |
Description: production-ready FastAPI microservice
```

**`summary_full`** — Used as BERT input. A full paragraph combining all signals with natural language context:
```
Repository: user/project. Description: production-ready FastAPI microservice with auth...
Primary language: Python. Age: 2y 1m old. Activity: actively maintained (pushed within last month).
Community: 280 stars, 42 forks. Team: 8 contributors. Development: 320 commits in the last year...
```

**Justification**: Structured text lets the LLM and BERT reason about relationships between signals rather than processing raw numbers, which improves classification accuracy and prompt reliability.

---

## Prompt Design

**Model**: Gemini 1.5 Flash (free tier)

**Prompt structure**:
1. System instruction with task description
2. Explicit definitions for all 6 categories
3. 6 few-shot examples (one per category) with realistic repo summaries
4. The repository summary to classify
5. Strict output format: one label only, no explanation

**Temperature**: 0.1 (low for consistency)

**Rationale**: Few-shot examples anchor the LLM to our definitions and prevent hallucination. Low temperature ensures reproducibility. Structured summaries give the LLM exactly the signals it needs, avoiding it making up information.

**Limitations**:
- LLM cannot read actual code
- Boundary cases (intern vs junior) are inherently ambiguous with metadata
- LLM may over-weight star count
- Gemini free tier: 15 RPM, so labeling 600 repos takes ~40 minutes

---

## Dataset Split

| Split | Size | Purpose |
|---|---|---|
| Train | 70% | Fine-tuning DistilBERT |
| Validation | 15% | Hyperparameter selection, early stopping |
| Test | 15% | Final evaluation (held out during training) |

Split is **stratified by label** to maintain class proportions.

---

## BERT Model Used

**Model**: `distilbert-base-uncased`

**Why DistilBERT**:
- 40% smaller and 60% faster than BERT-base
- Retains 97% of BERT's performance on classification tasks
- Fits in free Colab GPU (T4 16GB) and local CPU without issues
- Suitable for academic projects without large compute budgets

**Training configuration**:
- Epochs: 3
- Batch size: 16
- Learning rate: 2e-5 (AdamW)
- Max token length: 512
- Warmup: 10% of total steps
- Gradient clipping: 1.0

---

## Final Metrics

Metrics are saved to `output/metrics/` after running `evaluation.py`.

Expected ranges based on dataset composition:

| Model | Accuracy | F1 Macro |
|---|---|---|
| TF-IDF + LR (baseline) | ~0.65–0.72 | ~0.60–0.68 |
| DistilBERT | ~0.74–0.82 | ~0.70–0.78 |

Hardest categories to separate: `intern` vs `junior` (metadata signals are similar).
Easiest categories: `low_value` and `lead` (extreme signal profiles).

---

## Main Limitations

1. Cannot read actual code — classification relies entirely on metadata
2. Private repositories are invisible — developers may hide their best work
3. Star/fork counts favor popular ecosystems (Python, JavaScript) over niche ones
4. LLM labels are noisy — this is weak supervision, not ground truth
5. The system scores repositories, not developers — a person may have repos across multiple levels
6. Time-sensitive: a repository's classification may change as it grows

---

## Commercial Applications

- **Recruiters**: Filter thousands of GitHub profiles in seconds
- **Engineering managers**: Pre-screen candidates before technical interviews
- **Startups**: Identify high-quality open source contributors for hiring
- **Accelerators**: Technical due diligence on founders' GitHub portfolios
- **Platforms**: GitHub profile scoring tools (similar to CodersRank, GitStar)

---

## How to Run the Project

### 1. Setup

```bash
git clone https://github.com/YOUR_USERNAME/github_hiring_repository_intelligence
cd github_hiring_repository_intelligence
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the root directory:

```
GITHUB_TOKEN=ghp_your_token_here
GEMINI_API_KEY=your_gemini_key_here
```

**GitHub token**: github.com → Settings → Developer Settings → Personal Access Tokens → Classic → scope: `public_repo`

**Gemini API key**: aistudio.google.com → Get API Key

### 3. Run the pipeline (in order)

```bash
# Stage 1: Collect data from GitHub (~30-60 min depending on repo count)
python src/github_collector.py

# Stage 2: Clean and engineer features
python src/preprocessing.py

# Stage 2b: Generate text summaries
python src/summarization.py

# Stage 3: Label with Gemini (~40 min for 600 repos on free tier)
python src/llm_labeling.py

# Stage 4+5: Split data and fine-tune DistilBERT (~20-60 min depending on GPU)
python src/train.py

# Stage 6: Evaluate and generate figures
python src/evaluation.py
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## How to Run on Google Colab (free GPU)

1. Upload the project to Google Drive
2. Open a new Colab notebook
3. Mount Drive and install requirements:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/github_hiring_repository_intelligence
!pip install -r requirements.txt
```

4. Set environment variables and run `train.py`:

```python
import os
os.environ["GITHUB_TOKEN"] = "your_token"
os.environ["GEMINI_API_KEY"] = "your_key"
!python src/train.py
```

---

## GitHub Workflow

This project follows a feature-branch workflow:

```
main
├── feature/github-scraping      → github_collector.py
├── feature/preprocessing        → preprocessing.py + summarization.py
├── feature/llm-labeling         → llm_labeling.py
├── feature/bert-training        → train.py
├── feature/evaluation           → evaluation.py + visualization.py
└── feature/streamlit-dashboard  → app.py
```

All changes merged via Pull Requests into `main`.

---

## Video

See `video/link.txt` for the project presentation video link.
