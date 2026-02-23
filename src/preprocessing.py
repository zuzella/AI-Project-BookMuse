import re
import pandas as pd

TEXT_COLS_CANDIDATES = ["title", "authors", "categories", "description"]


def clean_text(x: str) -> str:
    """Basic cleaning: lowercase, remove HTML, keep letters/numbers, normalize spaces."""
    if pd.isna(x):
        return ""
    x = str(x).lower()
    x = re.sub(r"<[^>]+>", " ", x)          # remove html tags
    x = re.sub(r"[^a-z0-9\s]+", " ", x)     # keep letters/numbers/spaces
    x = re.sub(r"\s+", " ", x).strip()
    return x


def load_books(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Ensure expected columns exist
    for col in TEXT_COLS_CANDIDATES:
        if col not in df.columns:
            df[col] = ""

    # Clean text columns
    for col in TEXT_COLS_CANDIDATES:
        df[col] = df[col].apply(clean_text)

    # Combine into one text field used for vectorization
    df["combined_text"] = (
        df["title"] + " " + df["authors"] + " " + df["categories"] + " " + df["description"]
    ).str.strip()

    # For display and title matching
    df["display_title"] = df["title"].fillna("").astype(str)

    # Drop rows with no usable text
    df = df[df["combined_text"].str.len() > 0].reset_index(drop=True)

    return df