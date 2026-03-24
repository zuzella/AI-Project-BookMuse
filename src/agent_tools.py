"""
agent_tools.py
The 4 tools the AI agent can call, plus their JSON schemas.
"""

import pandas as pd

RATING_COLS = ["average_rating", "rating", "ratings"]
DESC_COLS = ["description", "desc", "summary"]
THUMB_COLS = ["thumbnail", "thumbnail_url", "image_url", "img_url", "cover"]


def _pick(row: dict, cols: list, default="") -> str:
    """Return first non-empty value found among the given column names."""
    for c in cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c])
    return default


def _summary(row: dict) -> dict:
    """Compact book dict sent back to GPT as tool result."""
    return {
        "title": row.get("display_title") or row.get("title", ""),
        "authors": row.get("authors", ""),
        "categories": row.get("categories", ""),
        "rating": _pick(row, RATING_COLS, "N/A"),
        "year": str(row.get("published_year", "")),
        "description": str(_pick(row, DESC_COLS, ""))[:300],
        "thumbnail": _pick(row, THUMB_COLS, ""),
    }


def search_books(query: str, df: pd.DataFrame, recommender, top_n: int = 8) -> list:
    """Free-text TF-IDF search over all books."""
    return [_summary(r) for r in recommender.recommend_by_text(query, top_n=top_n)]


def filter_books(
    df: pd.DataFrame,
    genre="",
    min_rating=0.0,
    max_rating=5.0,
    year_from=0,
    year_to=9999,
    author="",
    top_n=8,
) -> list:
    """Filter catalogue by genre, rating range, year range, or author."""
    r = df.copy()

    if genre and genre != "Any":
        r = r[r["categories"].fillna("").str.contains(genre, case=False, na=False)]

    if author:
        r = r[r["authors"].fillna("").str.contains(author, case=False, na=False)]

    rating_col = next((c for c in RATING_COLS if c in r.columns), None)
    if rating_col:
        nums = pd.to_numeric(r[rating_col], errors="coerce").fillna(0)
        r = r[(nums >= min_rating) & (nums <= max_rating)]

    if "published_year" in r.columns:
        years = pd.to_numeric(r["published_year"], errors="coerce").fillna(0)
        r = r[(years >= year_from) & (years <= year_to)]

    if rating_col:
        r = r.sort_values(rating_col, ascending=False)

    return [_summary(row) for _, row in r.head(top_n).iterrows()]


def recommend_books(title: str, recommender, top_n: int = 6) -> list:
    """Books similar to a given title."""
    return [_summary(r) for r in recommender.recommend_by_title(title, top_n=top_n)]


def get_book_info(title: str, df: pd.DataFrame) -> dict:
    """Full details for one specific book."""
    col = "display_title" if "display_title" in df.columns else "title"
    matches = df[df[col].fillna("").str.lower().str.contains(title.lower(), na=False)]

    if matches.empty:
        return {"error": f"No book found matching '{title}'"}

    row = matches.iloc[0].to_dict()
    return {
        "title": row.get(col, ""),
        "authors": row.get("authors", ""),
        "categories": row.get("categories", ""),
        "rating": _pick(row, RATING_COLS, "N/A"),
        "year": str(row.get("published_year", "")),
        "pages": str(row.get("num_pages", "")),
        "description": _pick(row, DESC_COLS, ""),
        "thumbnail": _pick(row, THUMB_COLS, ""),
    }


TOOL_SCHEMAS = [
    {
        "name": "search_books",
        "description": "Search books by topic, mood or theme using free text.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What the user is looking for"},
                "top_n": {"type": "integer", "default": 8},
            },
            "required": ["query"],
        },
    },
    {
        "name": "filter_books",
        "description": "Filter books by genre, rating range, year range, or author name.",
        "parameters": {
            "type": "object",
            "properties": {
                "genre": {"type": "string", "description": "Genre keyword e.g. Fiction"},
                "min_rating": {"type": "number", "default": 0},
                "max_rating": {"type": "number", "default": 5},
                "year_from": {"type": "integer", "default": 0},
                "year_to": {"type": "integer", "default": 9999},
                "author": {"type": "string", "description": "Author name keyword"},
                "top_n": {"type": "integer", "default": 8},
            },
            "required": [],
        },
    },
    {
        "name": "recommend_books",
        "description": "Recommend books similar to a title the user already knows.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Book title to base recommendations on"},
                "top_n": {"type": "integer", "default": 6},
            },
            "required": ["title"],
        },
    },
    {
        "name": "get_book_info",
        "description": "Get full details about one specific book by title.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Book title to look up"},
            },
            "required": ["title"],
        },
    },
]
