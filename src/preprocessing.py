"""
preprocessing.py
Loads the CSV, cleans text, and creates the 'combined_text' column used by TF-IDF.
"""
import re
import pandas as pd

TEXT_COLS = ["title", "authors", "categories", "description"]


def clean_text(x) -> str:
    """Lowercase, strip HTML tags, keep only letters/numbers/spaces."""
    if pd.isna(x):
        return ""
    x = str(x).lower()
    x = re.sub(r"<[^>]+>", " ", x)       # remove HTML tags
    x = re.sub(r"[^a-z0-9\s]+", " ", x)  # remove punctuation
    x = re.sub(r"\s+", " ", x).strip()
    return x


def load_books(csv_path: str) -> pd.DataFrame:
    """Read CSV, clean text columns, build combined_text for vectorization."""
    df = pd.read_csv(csv_path)

    for col in TEXT_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].apply(clean_text)

    # One big text string per book: title + authors + categories + description
    df["combined_text"] = (
        df["title"] + " " + df["authors"] + " " +
        df["categories"] + " " + df["description"]
    ).str.strip()

    df["display_title"] = df["title"].fillna("").astype(str)
    df = df[df["combined_text"].str.len() > 0].reset_index(drop=True)
    return df
