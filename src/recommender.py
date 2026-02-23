from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class BookRecommender:
    df: pd.DataFrame
    vectorizer: TfidfVectorizer
    tfidf_matrix: Any  # sparse matrix

    @classmethod
    def build(cls, df: pd.DataFrame) -> "BookRecommender":
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
        )
        tfidf_matrix = vectorizer.fit_transform(df["combined_text"])
        return cls(df=df, vectorizer=vectorizer, tfidf_matrix=tfidf_matrix)

    def recommend_by_title(self, title_query: str, top_n: int = 5) -> List[Dict[str, Any]]:
        title_query = (title_query or "").strip().lower()
        if not title_query:
            return []

        titles = self.df["display_title"].fillna("").astype(str)
        titles_low = titles.str.lower()

        exact = self.df[titles_low == title_query]
        if len(exact) > 0:
            idx = exact.index[0]
        else:
            contains = self.df[titles_low.str.contains(title_query, na=False)]
            if len(contains) == 0:
                return []
            idx = contains.index[0]

        sims = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        sims[idx] = -1  # exclude the same book

        best_idx = np.argsort(sims)[::-1][: int(top_n)]

        results = []
        for i in best_idx:
            row = self.df.iloc[i].to_dict()
            row["similarity"] = float(sims[i])
            results.append(row)
        return results