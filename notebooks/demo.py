"""
Movie Recommender — Demo Notebook
==================================

Run this end-to-end to see the recommender in action.
You can also copy each section into a Jupyter notebook cell.

Dataset: TMDB 5000 Movies
    https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
    Files needed: tmdb_5000_movies.csv, tmdb_5000_credits.csv
    Place them in the ./data/ folder.
"""

import sys
from pathlib import Path

# Make src importable.
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from preprocess import load_and_clean
from recommender import MovieRecommender


# ----------------------------------------------------------------------
# 1. Load and clean the data
# ----------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

movies = load_and_clean(
    movies_path=DATA_DIR / "tmdb_5000_movies.csv",
    credits_path=DATA_DIR / "tmdb_5000_credits.csv",
)

print(f"Loaded {len(movies):,} movies")
print(f"Columns available: {list(movies.columns)[:10]}...")


# ----------------------------------------------------------------------
# 2. Build the recommender
# ----------------------------------------------------------------------
feature_cols = [
    "genres_clean",
    "keywords_clean",
    "cast_clean",
    "director_clean",
    "overview",
]

rec = MovieRecommender(movies)
rec.fit(feature_cols)
print(f"\nTF-IDF matrix shape: {rec.tfidf_matrix.shape}")


# ----------------------------------------------------------------------
# 3. Try it out!
# ----------------------------------------------------------------------
test_titles = [
    "The Dark Knight",
    "Inception",
    "Toy Story",
    "The Avengers",
    "Pulp Fiction",
]

for title in test_titles:
    print(f"\n{'=' * 60}")
    print(f"Top 5 movies similar to: {title}")
    print("=" * 60)
    try:
        recommendations = rec.recommend(title, top_n=5)
        print(recommendations.to_string(index=False))
    except ValueError as e:
        print(e)

# ======================================================================
# BONUS: Compare different feature combinations
# ======================================================================
# Goal: See how recommendations change when we use different feature sets.
# This helps us understand which features carry the most signal.

print("\n\n")
print("#" * 70)
print("# FEATURE COMBINATION COMPARISON")
print("#" * 70)

# Define three different recommenders, each fit on different features
print("\nBuilding 3 recommenders with different feature sets...")

# Recommender 1: Genres only
rec_genres = MovieRecommender(movies)
rec_genres.fit(["genres_clean"])

# Recommender 2: Genres + overview text
rec_combo = MovieRecommender(movies)
rec_combo.fit(["genres_clean", "overview"])

# Recommender 3: All features (genres + keywords + cast + director + overview)
rec_all = MovieRecommender(movies)
rec_all.fit(["genres_clean", "keywords_clean", "cast_clean", "director_clean", "overview"])

print("Done!\n")

# Pick a movie to test all three on
test_movie = "The Matrix"

# Loop through each recommender and print its top 3 recommendations
models = [
    ("1. Genres only", rec_genres),
    ("2. Genres + Overview", rec_combo),
    ("3. All features", rec_all),
]

for name, model in models:
    print(f"\n--- {name} ---")
    print(f"Top 3 recommendations for '{test_movie}':")
    try:
        result = model.recommend(test_movie, top_n=3)
        print(result[["title", "similarity_score"]].to_string(index=False))
    except ValueError as e:
        print(e)

print("\n" + "=" * 70)
print("Observation: Notice how recommendations change as we add more features.")
print("Genres-only finds generic action films. Adding cast/director makes")
print("recommendations more thematically specific to the same creators.")
print("=" * 70)