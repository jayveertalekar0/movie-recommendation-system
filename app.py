import streamlit as st
import pandas as pd
import pickle
import requests
import difflib
import random

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# ------------------ TMDb Setup ------------------ #
TMDB_API_KEY = "4f08f66bf5057ce8a548d5271a09e5d4"  # 🔑 Replace with your TMDb API key
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# TMDb Genre IDs
GENRE_OPTIONS = {
    "Action": 28,
    "Adventure": 12,
    "Comedy": 35,
    "Romance": 10749,
    "Drama": 18,
    "Science Fiction": 878,
    "Horror": 27,
}

# ------------------ Fetch Movie Data ------------------ #
@st.cache_data
def fetch_movie_data(title=None, tmdb_id=None):
    try:
        if tmdb_id:
            url = f"{TMDB_BASE_URL}/movie/{tmdb_id}?api_key={TMDB_API_KEY}&language=en-US"
            response = requests.get(url)
            data = response.json()
        elif title:
            search_url = f"{TMDB_BASE_URL}/search/movie?api_key={TMDB_API_KEY}&query={title}"
            response = requests.get(search_url)
            results = response.json().get("results", [])
            if not results:
                return None
            data = results[0]
        else:
            return None

        poster = data.get("poster_path")
        return {
            "poster": f"{TMDB_IMAGE_BASE}{poster}" if poster else None,
            "title": data.get("title"),
            "year": data.get("release_date", "")[:4] if data.get("release_date") else "",
            "genre": ", ".join([g["name"] for g in data.get("genres", [])]) if data.get("genres") else "",
            "rating": data.get("vote_average"),
            "plot": data.get("overview"),
            "tmdb_id": data.get("id"),
            "imdb_id": data.get("imdb_id") if "imdb_id" in data else None,
        }
    except Exception as e:
        print(f"Error fetching movie data for {title or tmdb_id}: {e}")
    return None

# ------------------ Fetch Movies by Genre ------------------ #
@st.cache_data
def get_movies_by_genre(genre_id, n=20):
    url = f"{TMDB_BASE_URL}/discover/movie?api_key={TMDB_API_KEY}&with_genres={genre_id}&sort_by=popularity.desc"
    response = requests.get(url).json()
    results = response.get("results", [])[:n]
    movies = []
    for movie in results:
        movies.append({
            "title": movie["title"],
            "movie_data": {
                "poster": f"{TMDB_IMAGE_BASE}{movie['poster_path']}" if movie.get("poster_path") else None,
                "title": movie["title"],
                "year": movie.get("release_date", "")[:4],
                "rating": movie.get("vote_average"),
                "plot": movie.get("overview"),
                "tmdb_id": movie.get("id"),
            }
        })
    return movies

# ------------------ Load Pickle ------------------ #
@st.cache_resource
def load_data():
    try:
        with open("movie_recommender.pkl", "rb") as f:
            data = pickle.load(f)
        return (
            data["Required_movies"],
            data["language_models"],
            data["language_tfidf_matrices"],
            data["language_indices"]
        )
    except FileNotFoundError:
        st.error("The 'movie_recommender.pkl' file was not found. Please ensure it's in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        st.stop()

Required_movies, language_models, language_tfidf_matrices, language_indices = load_data()

# ------------------ Fuzzy Matching ------------------ #
def find_movies_by_title(user_input):
    exact_matches = Required_movies[Required_movies['title'].str.lower() == user_input.lower()]
    if not exact_matches.empty:
        return exact_matches['title'].unique().tolist()
    titles = Required_movies['title'].dropna().tolist()
    matches = difflib.get_close_matches(user_input.lower(), [t.lower() for t in titles], n=5, cutoff=0.6)
    matched_titles = []
    for match in matches:
        original_title = Required_movies[Required_movies['title'].str.lower() == match]['title'].iloc[0]
        if original_title not in matched_titles:
            matched_titles.append(original_title)
    return matched_titles

# ------------------ Recommend Movies ------------------ #
def recommend(movie_title, top_n=5):
    movie_row = Required_movies[Required_movies['title'].str.lower() == movie_title.lower()]
    if movie_row.empty:
        return []
    movie_lang = movie_row.iloc[0]['full_language']
    lang_df = language_indices[movie_lang]
    try:
        movie_idx_in_lang_df = lang_df[lang_df['title'].str.lower() == movie_title.lower()].index[0]
    except IndexError:
        return []
    nn_model = language_models[movie_lang]
    tfidf_matrix = language_tfidf_matrices[movie_lang]
    movie_vector = tfidf_matrix[movie_idx_in_lang_df]
    distances, indices = nn_model.kneighbors(movie_vector.reshape(1, -1), n_neighbors=top_n + 1)
    similar_indices = indices.flatten()[1:]
    recs = []
    for idx in similar_indices:
        row = lang_df.iloc[idx]
        recs.append({
            "title": row['title'],
            "movie_data": fetch_movie_data(title=row['title']),
        })
    return recs

# ------------------ Popular Movies ------------------ #
def get_popular_movies_by_language(lang, n=10):
    lang_movies = Required_movies[Required_movies['full_language'].str.contains(lang, case=False, na=False)].dropna(
        subset=['title'])
    top_n_titles = lang_movies['title'].head(n).tolist()
    popular_movies = []
    for title in top_n_titles:
        data = fetch_movie_data(title=title)
        if data:
            popular_movies.append({"title": title, "movie_data": data})
    return popular_movies

@st.cache_data
def get_shuffled_popular_movies(total_per_language=3, total_movies_to_display=12):
    all_popular_movies = []
    language_options = {
        "English Movies": "en",
        "Hindi Movies": "hi",
        "Telugu Movies": "te",
        "Tamil Movies": "ta",
        "Marathi Movies": "mr",
        "Malayalam Movies": "ml",
        "Kannada Movies": "kn"
    }
    for lang_code in language_options.values():
        movies_by_lang = Required_movies[Required_movies['full_language'].str.contains(lang_code, case=False, na=False)].dropna(
            subset=['title']).head(total_per_language)['title'].tolist()
        for title in movies_by_lang:
            data = fetch_movie_data(title=title)
            if data:
                all_popular_movies.append({"title": title, "movie_data": data})
    random.shuffle(all_popular_movies)
    return all_popular_movies[:total_movies_to_display]

# ------------------ UI Styling ------------------ #
st.markdown("""
<style>
.poster-container {
    position: relative;
    text-align: center;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
    width: 200px;
    margin: auto;
}
.poster-container:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 15px rgba(0,0,0,0.2);
}
.poster-container img {
    width: 200px;
    height: 300px;
    object-fit: cover;
    border-bottom: 1px solid #ddd;
}
.poster-title {
    font-weight: bold;
    margin-top: 5px;
    font-size: 16px;
}
.poster-plot {
    font-size: 13px;
    margin-top: 5px;
    color: #555;
    height: 60px;
    overflow: hidden;
    text-overflow: ellipsis;
}
.poster-container a {
    text-decoration: none;
    color: #FF4B4B;
    font-weight: bold;
    display: block;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ Display Card with Description ------------------ #
def display_movie_card(movie):
    st.markdown('<div class="poster-container">', unsafe_allow_html=True)
    if movie['movie_data'] and movie['movie_data'].get("poster"):
        st.image(movie['movie_data']["poster"])
    else:
        st.markdown('<div style="height: 300px; display: flex; align-items: center; justify-content: center; border: 1px dashed #ccc; background:#f0f2f6;">No Poster</div>', unsafe_allow_html=True)
    # Title
    if movie['movie_data'] and movie['movie_data'].get("title"):
        st.markdown(f'<div class="poster-title">{movie["movie_data"]["title"]} ({movie["movie_data"].get("year","")})</div>', unsafe_allow_html=True)
    # Plot
    if movie['movie_data'] and movie['movie_data'].get("plot"):
        st.markdown(f'<div class="poster-plot">{movie["movie_data"]["plot"]}</div>', unsafe_allow_html=True)
    # Links
    imdb_id = movie['movie_data'].get("imdb_id")
    if imdb_id:
        st.markdown(f'<a href="https://www.imdb.com/title/{imdb_id}/" target="_blank">IMDb Link</a>', unsafe_allow_html=True)
    elif movie['movie_data'].get("tmdb_id"):
        st.markdown(f'<a href="https://www.themoviedb.org/movie/{movie["movie_data"]["tmdb_id"]}" target="_blank">TMDb Link</a>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Session State ------------------ #
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'show_popular_movies_by_language' not in st.session_state:
    st.session_state.show_popular_movies_by_language = False
if 'show_genre_movies' not in st.session_state:
    st.session_state.show_genre_movies = False

# ------------------ Main ------------------ #
st.title("🎬 Movie Recommendation System (TMDb)")
st.subheader("🔍 Search for a Movie")
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("Enter a movie name:", key="user_input", label_visibility="collapsed")
with col2:
    search_button = st.button("Search", key="search_button")

if search_button:
    st.session_state.show_results = True
    st.session_state.show_popular_movies_by_language = False
    st.session_state.show_genre_movies = False
    matched_titles = find_movies_by_title(st.session_state.user_input)
    if matched_titles:
        st.session_state.matched_titles = matched_titles
    else:
        st.warning("No close match found. Try another title.")
        st.session_state.show_results = False
        if 'matched_titles' in st.session_state:
            del st.session_state['matched_titles']
    st.rerun()

# ------------------ Sidebar ------------------ #
st.sidebar.title("Browse Movies")
language_options = {
    "All Languages": "all",
    "English Movies": "en",
    "Hindi Movies": "hi",
    "Telugu Movies": "te",
    "Tamil Movies": "ta",
    "Marathi Movies": "mr",
    "Malayalam Movies": "ml",
    "Kannada Movies": "kn"
}

def handle_language_selection():
    if st.session_state.language_choice != "All Languages":
        st.session_state.show_popular_movies_by_language = True
        st.session_state.show_results = False
        st.session_state.show_genre_movies = False
    else:
        st.session_state.show_popular_movies_by_language = False

choice = st.sidebar.selectbox("Select a language:", list(language_options.keys()), key='language_choice', on_change=handle_language_selection)

# Genre Section
def handle_genre_selection():
    st.session_state.show_genre_movies = True
    st.session_state.show_results = False
    st.session_state.show_popular_movies_by_language = False

genre_choice = st.sidebar.selectbox("Select a genre:", list(GENRE_OPTIONS.keys()), key='genre_choice', on_change=handle_genre_selection)

# ------------------ Display Sections ------------------ #
if st.session_state.show_popular_movies_by_language:
    lang_code = language_options[st.session_state.language_choice]
    st.subheader(f"🎥 Top {st.session_state.language_choice}")
    movies = get_popular_movies_by_language(lang_code, n=15)
    if movies:
        cols = st.columns(5)
        for i, m in enumerate(movies):
            with cols[i % 5]:
                display_movie_card(m)
    else:
        st.warning(f"No movies for {st.session_state.language_choice}.")

elif st.session_state.show_genre_movies:
    st.subheader(f"🎭 Top {st.session_state.genre_choice} Movies")
    movies = get_movies_by_genre(GENRE_OPTIONS[st.session_state.genre_choice], n=20)
    if movies:
        cols = st.columns(5)
        for i, m in enumerate(movies):
            with cols[i % 5]:
                display_movie_card(m)
    else:
        st.warning(f"No movies found for {st.session_state.genre_choice}.")

elif st.session_state.show_results:
    matched_titles = st.session_state.matched_titles
    st.markdown("---")
    if matched_titles:
        st.subheader(f"Found {len(matched_titles)} movie(s) for '{st.session_state.user_input}':")
        max_cols = 5
        cols = st.columns(min(len(matched_titles), max_cols))
        for j, title in enumerate(matched_titles):
            with cols[j % max_cols]:
                movie_data = fetch_movie_data(title=title)
                if movie_data:
                    display_movie_card({"title": title, "movie_data": movie_data})

        for title in matched_titles:
            st.markdown(f"### 🎬 Recommendations for: {title}")
            recs = recommend(title, top_n=5)
            if not recs:
                st.error(f"No recommendations found for {title}.")
            else:
                rec_cols = st.columns(len(recs))
                for l, r in enumerate(recs):
                    with rec_cols[l]:
                        display_movie_card(r)
            st.markdown("<hr>", unsafe_allow_html=True)

else:
    st.subheader("🔥 Top Popular Movies")
    popular_movies = get_shuffled_popular_movies()
    if popular_movies:
        cols_per_row = 4
        num_movies = len(popular_movies)
        for i in range(0, num_movies, cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                movie_index = i + j
                if movie_index < num_movies:
                    movie_to_display = popular_movies[movie_index]
                    with cols[j]:
                        display_movie_card(movie_to_display)
    else:
        st.warning("No popular movies found.")
