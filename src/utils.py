"""
utils.py - Helper functions for the GitHub Hiring Repository Intelligence project.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


# ── Path helpers ──────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent

PATHS = {
    "raw":        BASE_DIR / "data" / "raw",
    "processed":  BASE_DIR / "data" / "processed",
    "labeled":    BASE_DIR / "data" / "labeled",
    "splits":     BASE_DIR / "data" / "splits",
    "models":     BASE_DIR / "models" / "trained_models",
    "figures":    BASE_DIR / "output" / "figures",
    "tables":     BASE_DIR / "output" / "tables",
    "metrics":    BASE_DIR / "output" / "metrics",
}

def ensure_dirs():
    for p in PATHS.values():
        p.mkdir(parents=True, exist_ok=True)


# ── JSON I/O ──────────────────────────────────────────────────────────────────

def save_json(data, filepath: Path):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(filepath: Path):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Environment ───────────────────────────────────────────────────────────────

def get_github_token() -> str:
    token = os.getenv("GITHUB_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "GITHUB_TOKEN not found. Add it to your .env file:\n  GITHUB_TOKEN=ghp_..."
        )
    return token


def get_gemini_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found. Add it to your .env file:\n  GEMINI_API_KEY=..."
        )
    return key


# ── Category definitions ──────────────────────────────────────────────────────

CATEGORIES = [
    "intern",
    "junior",
    "senior",
    "lead",
    "template_or_clone",
    "low_value",
]

CATEGORY_DESCRIPTIONS = {
    "intern":          "Intern-level: simple scripts, no structure, no tests, no docs",
    "junior":          "Junior-level: basic project structure, some docs, minimal tests",
    "senior":          "Senior-level: well structured, CI/CD, tests, clear documentation",
    "lead":            "Lead/Architect-level: complex architecture, multiple contributors, releases, design patterns",
    "template_or_clone": "Template or clone: boilerplate, forked with no meaningful changes",
    "low_value":       "Low value: empty, abandoned, or trivial repository",
}


# ── Misc ──────────────────────────────────────────────────────────────────────

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
