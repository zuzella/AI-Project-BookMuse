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
    tfidf_matrix: Any

    @classmethod
    def build(cls, df: pd.DataFrame) -> "BookRecommender":
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=50000)
        tfidf_matrix = vectorizer.fit_transform(df["combined_text"])
        return cls(df=df, vectorizer=vectorizer, tfidf_matrix=tfidf_matrix)

    def _find_title_index(self, title_query: str) -> Optional[int]:
        q = (title_query or "").strip().lower()
        if not q:
            return None

        titles = self.df["display_title"].fillna("").astype(str)
        titles_low = titles.str.lower()

        exact = self.df[titles_low == q]
        if len(exact) > 0:
            return int(exact.index[0])

        contains = self.df[titles_low.str.contains(q, na=False)]
        if len(contains) == 0:
            return None
        return int(contains.index[0])

    def _rank(self, query_vec, top_n: int, exclude_idx: Optional[int] = None) -> List[Dict[str, Any]]:
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        if exclude_idx is not None:
            sims[exclude_idx] = -1
        best = np.argsort(sims)[::-1][: int(top_n)]

        out: List[Dict[str, Any]] = []
        for i in best:
            row = self.df.iloc[i].to_dict()
            row["similarity"] = float(sims[i])
            out.append(row)
        return out

    def recommend_by_title(self, title_query: str, top_n: int = 6) -> List[Dict[str, Any]]:
        idx = self._find_title_index(title_query)
        if idx is None:
            return []
        return self._rank(self.tfidf_matrix[idx], top_n=top_n, exclude_idx=idx)

    def recommend_by_text(self, free_text: str, top_n: int = 6) -> List[Dict[str, Any]]:
        text = (free_text or "").strip()
        if not text:
            return []
        qv = self.vectorizer.transform([text])
        return self._rank(qv, top_n=top_n)