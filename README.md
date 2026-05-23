# 🎬 Content-Based Movie Recommender System

A Python implementation of a content-based recommender system that suggests movies similar to a given title using **TF-IDF** vectorization and **cosine similarity**. Built on the TMDB 5000 Movies dataset (~4,800 films).

> **Example output** — recommendations for *Inception*:
> 1. Interstellar (similarity: 0.41)
> 2. The Prestige (similarity: 0.36)
> 3. Memento (similarity: 0.33)
> 4. Source Code (similarity: 0.29)
> 5. Looper (similarity: 0.27)

---

## 📋 Project Overview

Recommender systems power much of the modern web — Netflix, Spotify, Amazon, YouTube. This project implements one of the two foundational approaches: **content-based filtering**, which recommends items based on their intrinsic features rather than user behavior.

Given a movie title, the system returns the most similar titles in the catalog based on a combination of:
- **Genres** (e.g. action, sci-fi, drama)
- **Plot keywords** (e.g. dream, time-travel, heist)
- **Top 3 cast members**
- **Director**
- **Plot overview** (free text)

---

## 🛠️ How It Works

The pipeline has four stages:

### 1. Data Preprocessing
The TMDB dataset stores `genres`, `keywords`, `cast`, and `crew` as stringified JSON. These are parsed into clean, space-separated strings. Multi-word entities (e.g. *"Science Fiction"*) are collapsed into single tokens (`sciencefiction`) so the vectorizer treats them as one concept.

### 2. Feature Engineering — "Soup" of Words
All cleaned text columns are concatenated into a single string per movie. This combined representation lets one model capture genre, theme, talent, and plot signals simultaneously.

### 3. TF-IDF Vectorization
Each movie's text soup is converted into a high-dimensional vector using **Term Frequency–Inverse Document Frequency**. TF-IDF down-weights common words (*"the"*, *"man"*) and up-weights distinctive ones (*"cyberpunk"*, *"nolan"*) — so similarity is driven by what makes a film *distinctive*, not just frequent.

Bigrams `(1, 2)` are included to capture phrases like *"film noir"* or *"coming of age"*.

### 4. Cosine Similarity
Similarity between any two movies is computed as the cosine of the angle between their TF-IDF vectors. The result is a 4,800 × 4,800 matrix; for any query title, we sort its row to find the top-N closest neighbors.

---

## 📁 Project Structure

```
movie-recommender/
├── README.md                ← you are here
├── requirements.txt
├── data/                    ← place TMDB CSVs here (see Setup)
├── src/
│   ├── preprocess.py        ← parses raw TMDB JSON columns
│   └── recommender.py       ← MovieRecommender class
└── notebooks/
    └── demo.py              ← end-to-end usage example
```

---

## 🚀 Setup & Usage

### 1. Clone the repo and install dependencies
```bash
git clone https://github.com/<your-username>/movie-recommender.git
cd movie-recommender
pip install -r requirements.txt
```

### 2. Download the data
Grab the TMDB 5000 Movies dataset from Kaggle:
👉 https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Place these two files in `./data/`:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

### 3. Run the demo
```bash
python notebooks/demo.py
```

### 4. Use it in your own code
```python
from src.preprocess import load_and_clean
from src.recommender import MovieRecommender

movies = load_and_clean("data/tmdb_5000_movies.csv", "data/tmdb_5000_credits.csv")

rec = MovieRecommender(movies)
rec.fit(["genres_clean", "keywords_clean", "cast_clean", "director_clean", "overview"])

print(rec.recommend("The Dark Knight", top_n=10))
```

---

## 📊 Sample Results

| Query | Top 3 Recommendations |
|---|---|
| *The Dark Knight* | Batman Begins, The Dark Knight Rises, Batman |
| *Toy Story* | Toy Story 2, Toy Story 3, Monsters, Inc. |
| *Pulp Fiction* | Reservoir Dogs, Jackie Brown, Kill Bill: Vol. 1 |
| *The Avengers* | Avengers: Age of Ultron, Iron Man 2, Captain America: Civil War |

*(Exact rankings depend on the snapshot of the TMDB dataset used.)*

---

## 🧠 Design Choices & Trade-offs

**Why content-based and not collaborative filtering?**
Collaborative filtering needs user-rating data, which isn't included in this TMDB snapshot. Content-based also has the advantage of working on cold-start items (a brand-new film with no ratings can still be recommended on day one).

**Why TF-IDF instead of word embeddings (Word2Vec, BERT)?**
TF-IDF is interpretable, fast, and requires no GPU or pretrained model. For a catalog of ~5,000 movies the quality gap is small and the engineering complexity is much lower. Swapping in sentence embeddings is a natural next step (see below).

**Why concatenate features instead of weighting them separately?**
A single soup keeps the code simple and is competitive in practice. A weighted ensemble (e.g. 0.4·genre + 0.3·cast + 0.3·overview) would give finer control and is on the roadmap below.

---

## 🔭 Limitations & Future Work

- **No personalization.** The same query always returns the same list — there's no user profile. A hybrid system combining content-based and collaborative filtering (matrix factorization, ALS) would address this.
- **No semantic understanding.** TF-IDF treats *"dream"* and *"nightmare"* as unrelated tokens. Replacing the vectorizer with **sentence-transformers** embeddings would capture semantic similarity.
- **Cold catalog only.** Quality depends on metadata richness. Films with sparse overviews/keywords get weaker recommendations.
- **Evaluation is qualitative.** Without ground-truth user data, performance is judged by inspection. A natural extension is to score recommendations against MovieLens ratings using *precision@k*.

---

## 📚 Tech Stack
- **Python 3.10+**
- **pandas** — data manipulation
- **scikit-learn** — TF-IDF and cosine similarity
- **numpy** — numerical operations

---

## 📄 License
MIT — free to use, modify, and distribute.

---

*Built as a portfolio project to demonstrate practical NLP, feature engineering, and clean software design in Python.*
