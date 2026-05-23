"""
Streamlit web app for the content-based movie recommender.
Run with:  streamlit run app.py
"""

import sys
from pathlib import Path

# Make the src/ folder importable
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import streamlit as st
import pandas as pd

from preprocess import load_and_clean
from recommender import MovieRecommender


# ----------------------------------------------------------------------
# Page configuration — this MUST be the first Streamlit command
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",                    # Use full screen width
    initial_sidebar_state="expanded", # Show sidebar by default
)

# ----------------------------------------------------------------------
# Cached data loading and model building
# ----------------------------------------------------------------------
# @st.cache_data tells Streamlit: "run this function ONCE, then reuse
# the result every time the script reruns." Without this, we'd reload
# the CSVs and rebuild the TF-IDF matrix on every keystroke (slow!).
# ----------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"


@st.cache_data(show_spinner="Loading movie data...")
def load_movies():
    """Load and preprocess the TMDB dataset. Cached so it only runs once."""
    return load_and_clean(
        movies_path=str(DATA_DIR / "tmdb_5000_movies.csv"),
        credits_path=str(DATA_DIR / "tmdb_5000_credits.csv"),
    )


@st.cache_resource(show_spinner="Building recommendation engine...")
def build_recommender(_movies_df):
    """
    Build the recommender. Uses @st.cache_resource (not cache_data)
    because the recommender holds a large numpy matrix that shouldn't
    be serialized.

    The leading underscore in _movies_df tells Streamlit not to try
    hashing it (DataFrames can be slow to hash).
    """
    feature_cols = [
        "genres_clean",
        "keywords_clean",
        "cast_clean",
        "director_clean",
        "overview",
    ]
    rec = MovieRecommender(_movies_df)
    rec.fit(feature_cols)
    return rec


# Load data and build recommender (only runs once thanks to caching)
movies = load_movies()
recommender = build_recommender(movies)


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🎬 Movie Recommender")
st.markdown(
    "Find your favorite movies"
)
st.divider()


# ----------------------------------------------------------------------
# Sidebar — controls live here so they don't crowd the main area
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    num_recommendations = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=20,
        value=10,
        help="How many similar movies to return",
    )

    st.divider()
    st.markdown("### About")
    st.markdown(
        "This app uses **TF-IDF vectorization** to convert movie metadata "
        "(genres, keywords, cast, director, plot) into numerical vectors, "
        "then computes **cosine similarity** between every pair of movies."
    )
    st.markdown(f"📊 **{len(movies):,}** movies in the catalog")

    # ----------------------------------------------------------------------
# Main area: movie search
# ----------------------------------------------------------------------
st.subheader("🔍 Search for a movie")

# Build a sorted list of all movie titles for the dropdown
all_titles = sorted(movies["title"].dropna().unique().tolist())

selected_movie = st.selectbox(
    "Pick a movie from the catalog (or start typing to search):",
    options=all_titles,
    index=all_titles.index("The Dark Knight") if "The Dark Knight" in all_titles else 0,
    help="Type to filter — the dropdown supports search",
)


# ----------------------------------------------------------------------
# Recommend button + results
# ----------------------------------------------------------------------
if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
    with st.spinner(f"Finding movies similar to '{selected_movie}'..."):
        try:
            results = recommender.recommend(selected_movie, top_n=num_recommendations)
        except ValueError as e:
            st.error(str(e))
            st.stop()

    st.success(f"Found {len(results)} similar movies!")
    st.divider()

    # Display each recommendation as a styled card
    st.subheader(f"📽️ Movies similar to *{selected_movie}*")

    for i, row in results.iterrows():
        with st.container(border=True):
            cols = st.columns([3, 1, 1])

            with cols[0]:
                st.markdown(f"### {i + 1}. {row['title']}")
                if "genres" in results.columns and pd.notna(row.get("genres")):
                    # genres is still raw JSON-ish text — extract names manually
                    import ast
                    try:
                        genres_list = ast.literal_eval(row["genres"])
                        genre_names = [g["name"] for g in genres_list if isinstance(g, dict)]
                        if genre_names:
                            st.caption(" · ".join(genre_names))
                    except (ValueError, SyntaxError):
                        pass

            with cols[1]:
                if "vote_average" in results.columns and pd.notna(row.get("vote_average")):
                    st.metric("Rating", f"⭐ {row['vote_average']:.1f}")

            with cols[2]:
                st.metric("Similarity", f"{row['similarity_score']:.2f}")
                # Visual progress bar of similarity
                st.progress(min(float(row["similarity_score"]), 1.0))


# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.divider()
st.caption(
    "Built with Python, scikit-learn, and Streamlit · "
    "Data: TMDB 5000 Movies Dataset (Kaggle)"
)