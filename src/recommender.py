"""
Content-Based Movie Recommender System
---------------------------------------
This module builds a content-based recommendation engine that suggests
movies similar to a given title based on genres, keywords, cast, and overview.

Approach:
    1. Combine relevant text features (genres, keywords, overview) into a
       single "soup" of words for each movie.
    2. Convert the text into a TF-IDF feature matrix.
    3. Compute cosine similarity between every pair of movies.
    4. For a given movie, return the top-N most similar titles.

Author: [Your Name]
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    """A content-based movie recommender using TF-IDF and cosine similarity."""

    def __init__(self, movies_df: pd.DataFrame):
        """
        Parameters
        ----------
        movies_df : pd.DataFrame
            DataFrame containing at minimum a 'title' column and one or more
            text columns that describe each movie (e.g. genres, overview).
        """
        self.movies = movies_df.reset_index(drop=True).copy()
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None

    @staticmethod
    def _clean_text(text: str) -> str:
        """Lowercase text and strip whitespace; handle missing values safely."""
        if pd.isna(text):
            return ""
        return str(text).lower().strip()

    def build_feature_soup(self, feature_cols: list) -> pd.Series:
        """
        Combine multiple text columns into a single string per movie.

        This 'soup' is what the TF-IDF vectorizer will operate on.
        """
        soup = self.movies[feature_cols].fillna("").astype(str).agg(" ".join, axis=1)
        return soup.apply(self._clean_text)

    def fit(self, feature_cols: list) -> None:
        """
        Build the TF-IDF matrix and cosine similarity matrix.

        Parameters
        ----------
        feature_cols : list of str
            Column names to combine into the text representation.
        """
        soup = self.build_feature_soup(feature_cols)

        # TF-IDF: down-weights common words, highlights distinctive ones.
        # Using bigrams (1, 2) captures pairs like "science fiction".
        tfidf = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=20000,
        )
        self.tfidf_matrix = tfidf.fit_transform(soup)

        # Cosine similarity between every pair of movies.
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        # Title -> row index lookup (lower-cased for case-insensitive search).
        self.indices = pd.Series(
            self.movies.index, index=self.movies["title"].str.lower()
        ).drop_duplicates()

    def recommend(self, title: str, top_n: int = 10) -> pd.DataFrame:
        """
        Return the top-N movies most similar to `title`.

        Parameters
        ----------
        title : str
            The movie title to base recommendations on (case-insensitive).
        top_n : int
            Number of recommendations to return.

        Returns
        -------
        pd.DataFrame with columns ['title', 'similarity_score', ...]
        """
        if self.cosine_sim is None:
            raise RuntimeError("Call .fit() before requesting recommendations.")

        key = title.lower().strip()
        if key not in self.indices:
            # Fuzzy hint: show titles that contain the search term.
            close = [
                t for t in self.indices.index if key in t
            ][:5]
            raise ValueError(
                f"'{title}' not found. Did you mean: {close}" if close
                else f"'{title}' not found in dataset."
            )

        idx = self.indices[key]
        if isinstance(idx, pd.Series):  # handles duplicate titles
            idx = idx.iloc[0]

        # Pair each movie with its similarity score, then sort.
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Skip index 0 — that's the movie itself.
        sim_scores = sim_scores[1: top_n + 1]
        movie_indices = [i for i, _ in sim_scores]
        scores = [round(s, 4) for _, s in sim_scores]

        result = self.movies.iloc[movie_indices][["title"]].copy()
        result["similarity_score"] = scores

        # Include extra context if those columns exist.
        for col in ("genres", "vote_average", "release_date"):
            if col in self.movies.columns:
                result[col] = self.movies.iloc[movie_indices][col].values

        return result.reset_index(drop=True)
