"""
github_collector.py - Stage 1: Collect repository data from the GitHub API.

Strategy:
- We query repositories across different search terms to get a diverse sample
  that spans all maturity levels (intern to lead, templates, low-value).
- We use the GitHub REST Search API (no GraphQL needed for our signals).
- Rate limit: 30 requests/min authenticated. We add small sleeps to stay safe.

Signals collected (>= 6 required):
  1.  stargazers_count
  2.  forks_count
  3.  open_issues_count
  4.  watchers_count
  5.  size (KB)
  6.  contributors_count  (extra API call per repo)
  7.  commit_count_last_year
  8.  has_wiki
  9.  has_projects
  10. has_discussions
  11. license
  12. topics
  13. default_branch
  14. created_at
  15. updated_at
  16. pushed_at
  17. has_readme         (contents API)
  18. readme_length      (proxy for documentation quality)
  19. has_ci_workflow    (contents API - .github/workflows)
  20. open_prs_count     (pulls API)
  21. release_count      (releases API)
  22. language
  23. description
  24. full_name
"""

import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent))
from utils import get_logger, get_github_token, PATHS, save_json

logger = get_logger("github_collector")

GITHUB_API = "https://api.github.com"

# ── Search queries designed to sample different maturity levels ───────────────
SEARCH_QUERIES = [
    # Senior / Lead quality
    ("stars:>500 pushed:>2024-01-01 language:Python",          80),
    ("stars:>500 pushed:>2024-01-01 language:TypeScript",       80),
    ("stars:>200 topic:machine-learning pushed:>2024-01-01",    60),
    ("stars:>200 topic:devops pushed:>2024-01-01",              60),
    # Junior / Intern quality
    ("stars:1..20 pushed:>2023-01-01 language:Python",          80),
    ("stars:1..20 pushed:>2023-01-01 language:JavaScript",      80),
    # Templates and clones
    ("topic:template stars:<50",                                 60),
    ("fork:true stars:0 pushed:<2023-01-01",                    60),
    # Low value / abandoned
    ("stars:0 pushed:<2022-01-01 size:<100",                    60),
    ("stars:0 size:1..50 language:Python",                      60),
]

HEADERS = {}   # populated in main()


def _get(url: str, params: dict = None, retries: int = 3) -> dict | list | None:
    """GET with retry logic."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 403:
                reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset - time.time(), 0) + 5
                logger.warning(f"Rate limited. Sleeping {wait:.0f}s …")
                time.sleep(wait)
            elif r.status_code == 404:
                return None
            else:
                logger.warning(f"HTTP {r.status_code} for {url}")
                time.sleep(2 ** attempt)
        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            time.sleep(2 ** attempt)
    return None


def _search_repos(query: str, per_page: int = 30, max_results: int = 100) -> list[dict]:
    """Search GitHub repos and return raw items."""
    items = []
    page = 1
    while len(items) < max_results:
        data = _get(
            f"{GITHUB_API}/search/repositories",
            params={"q": query, "per_page": per_page, "page": page, "sort": "updated"},
        )
        if not data or "items" not in data:
            break
        batch = data["items"]
        if not batch:
            break
        items.extend(batch)
        page += 1
        time.sleep(1.2)   # ~30 req/min safe zone
    return items[:max_results]


def _contributors_count(full_name: str) -> int:
    data = _get(
        f"{GITHUB_API}/repos/{full_name}/contributors",
        params={"per_page": 1, "anon": "true"},
    )
    # When there are many contributors GitHub returns a Link header with last page
    # We use a HEAD trick to get the count cheaply
    try:
        r = requests.get(
            f"{GITHUB_API}/repos/{full_name}/contributors",
            headers=HEADERS,
            params={"per_page": 1, "anon": "true"},
            timeout=10,
        )
        if r.status_code != 200:
            return 0
        link = r.headers.get("Link", "")
        if 'rel="last"' in link:
            import re
            match = re.search(r'page=(\d+)>; rel="last"', link)
            if match:
                return int(match.group(1))
        return len(r.json())
    except Exception:
        return 0


def _commit_count_last_year(full_name: str) -> int:
    data = _get(f"{GITHUB_API}/repos/{full_name}/stats/commit_activity")
    if not data:
        return 0
    try:
        return sum(week.get("total", 0) for week in data)
    except Exception:
        return 0


def _has_readme(full_name: str) -> tuple[bool, int]:
    data = _get(f"{GITHUB_API}/repos/{full_name}/readme")
    if not data:
        return False, 0
    size = data.get("size", 0)
    return True, size


def _has_ci_workflow(full_name: str) -> bool:
    data = _get(f"{GITHUB_API}/repos/{full_name}/contents/.github/workflows")
    return isinstance(data, list) and len(data) > 0


def _open_prs_count(full_name: str) -> int:
    r = requests.get(
        f"{GITHUB_API}/repos/{full_name}/pulls",
        headers=HEADERS,
        params={"state": "open", "per_page": 1},
        timeout=10,
    )
    if r.status_code != 200:
        return 0
    link = r.headers.get("Link", "")
    if 'rel="last"' in link:
        import re
        match = re.search(r'page=(\d+)>; rel="last"', link)
        if match:
            return int(match.group(1))
    return len(r.json())


def _release_count(full_name: str) -> int:
    r = requests.get(
        f"{GITHUB_API}/repos/{full_name}/releases",
        headers=HEADERS,
        params={"per_page": 1},
        timeout=10,
    )
    if r.status_code != 200:
        return 0
    link = r.headers.get("Link", "")
    if 'rel="last"' in link:
        import re
        match = re.search(r'page=(\d+)>; rel="last"', link)
        if match:
            return int(match.group(1))
    return len(r.json())


def enrich_repo(repo: dict) -> dict:
    """Add extra signals to a raw repo dict."""
    full_name = repo["full_name"]
    time.sleep(0.5)

    has_readme, readme_size = _has_readme(full_name)
    has_ci = _has_ci_workflow(full_name)
    contribs = _contributors_count(full_name)
    commits = _commit_count_last_year(full_name)
    prs = _open_prs_count(full_name)
    releases = _release_count(full_name)

    return {
        # Identity
        "full_name":            full_name,
        "description":          repo.get("description") or "",
        "language":             repo.get("language") or "Unknown",
        "topics":               repo.get("topics", []),
        "default_branch":       repo.get("default_branch", "main"),
        "license":              repo.get("license", {}).get("spdx_id") if repo.get("license") else None,
        # Dates
        "created_at":           repo.get("created_at"),
        "updated_at":           repo.get("updated_at"),
        "pushed_at":            repo.get("pushed_at"),
        # Basic signals
        "stargazers_count":     repo.get("stargazers_count", 0),
        "forks_count":          repo.get("forks_count", 0),
        "open_issues_count":    repo.get("open_issues_count", 0),
        "watchers_count":       repo.get("watchers_count", 0),
        "size_kb":              repo.get("size", 0),
        "is_fork":              repo.get("fork", False),
        "is_archived":          repo.get("archived", False),
        "has_wiki":             repo.get("has_wiki", False),
        "has_projects":         repo.get("has_projects", False),
        "has_discussions":      repo.get("has_discussions", False),
        # Enriched signals
        "contributors_count":   contribs,
        "commit_count_last_year": commits,
        "has_readme":           has_readme,
        "readme_size_bytes":    readme_size,
        "has_ci_workflow":      has_ci,
        "open_prs_count":       prs,
        "release_count":        releases,
    }


def collect(total_target: int = 600) -> pd.DataFrame:
    """
    Collect repositories from GitHub and save raw data.

    Args:
        total_target: approximate number of unique repos to collect

    Returns:
        DataFrame with all collected repos
    """
    global HEADERS
    token = get_github_token()
    HEADERS = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    seen = set()
    all_repos = []

    per_query = total_target // len(SEARCH_QUERIES) + 10

    for query, max_res in SEARCH_QUERIES:
        logger.info(f"Searching: {query!r}  (max={max_res})")
        raw_items = _search_repos(query, max_results=max_res)
        logger.info(f"  -> {len(raw_items)} raw results")
        for repo in raw_items:
            fn = repo["full_name"]
            if fn in seen:
                continue
            seen.add(fn)
            all_repos.append(repo)
        time.sleep(2)

    logger.info(f"Total unique repos before enrichment: {len(all_repos)}")

    # Enrich (this takes time — each repo = ~6 extra API calls)
    enriched = []
    for repo in tqdm(all_repos, desc="Enriching repos"):
        try:
            enriched.append(enrich_repo(repo))
        except Exception as e:
            logger.warning(f"Failed to enrich {repo.get('full_name')}: {e}")

    df = pd.DataFrame(enriched)
    out_path = PATHS["raw"] / "repositories_raw.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} repos to {out_path}")

    # Also save as JSON for inspection
    save_json(enriched, PATHS["raw"] / "repositories_raw.json")
    return df


if __name__ == "__main__":
    df = collect(total_target=600)
    print(df.shape)
    print(df.head())
