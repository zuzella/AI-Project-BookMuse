"""
recommender.py
Builds TF-IDF vectors for every book and finds similar ones via cosine similarity.

TF-IDF:  converts each book's text into a vector of numbers.
         Words that are unique to a book score high; common words score low.
Cosine similarity: measures the angle between two vectors.
         Score 1.0 = identical, 0.0 = completely unrelated.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class BookRecommender:
    df: pd.DataFrame
    vectorizer: TfidfVectorizer
    tfidf_matrix: Any  # sparse matrix: shape (num_books, num_unique_words)

    @classmethod
    def build(cls, df: pd.DataFrame) -> "BookRecommender":
        """Fit TF-IDF on all books. Called once at startup."""
        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),  # single words + two-word phrases
            min_df=2,
            max_features=50_000,
        )
        matrix = vec.fit_transform(df["combined_text"])
        return cls(df=df, vectorizer=vec, tfidf_matrix=matrix)

    def _find_index(self, title: str) -> Optional[int]:
        """Return the row index of a book by title (exact match first, then partial)."""
        q = title.strip().lower()
        titles = self.df["display_title"].str.lower()
        exact = self.df[titles == q]
        if not exact.empty:
            return int(exact.index[0])
        partial = self.df[titles.str.contains(q, na=False)]
        return int(partial.index[0]) if not partial.empty else None

    def _top_n(self, query_vec, top_n: int, exclude: Optional[int] = None) -> List[Dict[str, Any]]:
        """Compute cosine similarity and return top_n results as list of dicts."""
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        if exclude is not None:
            scores[exclude] = -1
        indices = np.argsort(scores)[::-1][:top_n]
        results = []
        for i in indices:
            row = self.df.iloc[i].to_dict()
            row["similarity"] = float(scores[i])
            results.append(row)
        return results

    def recommend_by_title(self, title: str, top_n: int = 6) -> List[Dict[str, Any]]:
        """Books similar to a given title."""
        idx = self._find_index(title)
        if idx is None:
            return []
        return self._top_n(self.tfidf_matrix[idx], top_n=top_n, exclude=idx)

    def recommend_by_text(self, text: str, top_n: int = 6) -> List[Dict[str, Any]]:
        """Books matching any free-text query (topic, mood, description)."""
        if not text.strip():
            return []
        vec = self.vectorizer.transform([text])
        return self._top_n(vec, top_n=top_n)
