"""
Data preprocessing for the TMDB 5000 Movies dataset.

The raw dataset stores some columns (genres, keywords, cast, crew) as
stringified JSON. This module parses those into clean strings the
recommender can use.
"""

import ast
import pandas as pd


def parse_json_column(value, key: str = "name", top_n: int | None = None) -> str:
    """
    Convert a stringified list of dicts into a space-separated string.

    Example
    -------
    Input:  "[{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}]"
    Output: "action adventure"
    """
    if pd.isna(value):
        return ""
    try:
        items = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return ""

    names = [str(item.get(key, "")) for item in items if isinstance(item, dict)]
    if top_n is not None:
        names = names[:top_n]

    # Remove spaces inside multi-word names so "Science Fiction" stays one token.
    names = [name.lower().replace(" ", "") for name in names if name]
    return " ".join(names)


def get_director(crew_value) -> str:
    """Extract director name from a stringified crew list."""
    if pd.isna(crew_value):
        return ""
    try:
        crew = ast.literal_eval(crew_value)
    except (ValueError, SyntaxError):
        return ""

    for member in crew:
        if isinstance(member, dict) and member.get("job") == "Director":
            return str(member.get("name", "")).lower().replace(" ", "")
    return ""


def load_and_clean(movies_path: str, credits_path: str | None = None) -> pd.DataFrame:
    """
    Load and merge TMDB 5000 movies + credits, returning a clean DataFrame
    ready for the recommender.
    """
    movies = pd.read_csv(movies_path)

    if credits_path:
        credits = pd.read_csv(credits_path)
        # TMDB dataset quirk: credits has 'movie_id', movies has 'id'.
        credits = credits.rename(columns={"movie_id": "id"})
        movies = movies.merge(credits, on="id", how="left", suffixes=("", "_dup"))
        # Drop duplicate title column from the merge.
        movies = movies.drop(columns=[c for c in movies.columns if c.endswith("_dup")])

    # Parse JSON-like columns.
    if "genres" in movies.columns:
        movies["genres_clean"] = movies["genres"].apply(parse_json_column)
    if "keywords" in movies.columns:
        movies["keywords_clean"] = movies["keywords"].apply(parse_json_column)
    if "cast" in movies.columns:
        # Only the top 3 billed actors carry signal.
        movies["cast_clean"] = movies["cast"].apply(
            lambda x: parse_json_column(x, top_n=3)
        )
    if "crew" in movies.columns:
        movies["director_clean"] = movies["crew"].apply(get_director)

    # Keep overview as-is (it's plain English already).
    movies["overview"] = movies["overview"].fillna("")

    # Drop rows missing a title (rare but possible).
    movies = movies.dropna(subset=["title"]).reset_index(drop=True)

    return movies
