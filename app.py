import streamlit as st
import pandas as pd
import pickle
import requests
import difflib
import random

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")


# ------------------ Fetch Poster + Info from OMDb ------------------ #
@st.cache_data
def fetch_movie_data(title=None, imdb_id=None):
    try:
        api_key = "9d4adebf"  # 🔑 Replace with your OMDb API key

        if imdb_id:
            url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={api_key}"
        elif title:
            url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
        else:
            return None

        response = requests.get(url)
        data = response.json()

        if data.get("Response") == "True" and data.get("Poster") != "N/A":
            return {
                "poster": data.get("Poster"),
                "title": data.get("Title"),
                "year": data.get("Year"),
                "genre": data.get("Genre"),
                "rating": data.get("imdbRating"),
                "plot": data.get("Plot"),
                "imdb_id": data.get("imdbID")
            }
    except Exception as e:
        print(f"Error fetching movie data for {title or imdb_id}: {e}")
    return None


# ------------------ Load Pickle ------------------ #
@st.cache_resource
def load_data():
    try:
        with open("movie_recommender.pkl", "rb") as f:
            data = pickle.load(f)
        return (
            data["Required_movies"],  # DataFrame
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


# ------------------ Fuzzy Matching & Recommendation Logic ------------------ #
def find_movies_by_title(user_input):
    """Finds all movies with titles that closely match the user's input."""
    # Exact matches first
    exact_matches = Required_movies[Required_movies['title'].str.lower() == user_input.lower()]
    if not exact_matches.empty:
        return exact_matches['title'].unique().tolist()

    # Fuzzy matching
    titles = Required_movies['title'].dropna().tolist()
    # Find unique, close matches from the full title list
    matches = difflib.get_close_matches(user_input.lower(), [t.lower() for t in titles], n=5, cutoff=0.6)

    # Map the lowercase matches back to their original case titles
    matched_titles = []
    for match in matches:
        # Find the original title for the lowercase match
        original_title = Required_movies[Required_movies['title'].str.lower() == match]['title'].iloc[0]
        # Check for duplicates before adding to the list
        if original_title not in matched_titles:
            matched_titles.append(original_title)
    return matched_titles


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
    sims = distances.flatten()[1:]
    recs = []
    for idx, score in zip(similar_indices, sims):
        row = lang_df.iloc[idx]
        recs.append({
            "title": row['title'],
            "score": 1 - float(score),
            "movie_data": fetch_movie_data(title=row['title']),
        })
    return recs


# ------------------ Popular Movies Section ------------------ #
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
        movies_by_lang = \
        Required_movies[Required_movies['full_language'].str.contains(lang_code, case=False, na=False)].dropna(
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
    width: 100%;
    text-align: center;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
}
.poster-container:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 15px rgba(0,0,0,0.2);
}
.poster-container .poster-image {
    width: 100%;
    height: auto;
    display: block;
    max-width: 200px;
    max-height: 300px;
    object-fit: cover;
    margin: 0 auto;
}
.poster-container a {
    text-decoration: none;
    color: #FF4B4B;
    font-weight: bold;
    display: block;
    margin-top: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ------------------ Display Movie Card ------------------ #
def display_movie_card(movie):
    st.markdown('<div class="poster-container">', unsafe_allow_html=True)
    if movie['movie_data'] and movie['movie_data'].get("poster") and movie['movie_data']["poster"] != "N/A":
        st.image(movie['movie_data']["poster"], use_container_width=False, output_format="JPEG",
                 caption=movie['movie_data']['title'])
    else:
        st.markdown(
            '<div style="height: 300px; display: flex; align-items: center; justify-content: center; border: 1px dashed #ccc; background-color: #f0f2f6;">No Poster Available</div>',
            unsafe_allow_html=True)

    if movie['movie_data'] and movie['movie_data'].get("imdb_id"):
        imdb_url = f"https://www.imdb.com/title/{movie['movie_data']['imdb_id']}/"
        st.markdown(f'<a href="{imdb_url}" target="_blank">Click here for IMDb</a>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# Initialize session state for controlling page sections
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'show_popular_movies_by_language' not in st.session_state:
    st.session_state.show_popular_movies_by_language = False

# ------------------ Main Content - Search Bar ------------------ #
st.title("🎬 Movie Recommendation System")
st.subheader("🔍 Search for a Movie")
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("Enter a movie name:", key="user_input", label_visibility="collapsed")
with col2:
    search_button = st.button("Search", key="search_button")

if search_button:
    # Clear other sections when a search is performed
    st.session_state.show_results = True
    st.session_state.show_popular_movies_by_language = False

    matched_titles = find_movies_by_title(st.session_state.user_input)
    if matched_titles:
        st.session_state.matched_titles = matched_titles
    else:
        st.warning("No close match found. Try a different title.")
        st.session_state.show_results = False
        if 'matched_titles' in st.session_state:
            del st.session_state['matched_titles']
    st.rerun()

# ------------------ Sidebar ------------------ #
st.sidebar.title("Browse Movies by Language")
language_options = {
    "All Languages": "all",
    "English Movies": "en",
    "Hindi Movies": "hi",
    "Telugu Movies": "te",
    "Tamil Movies": "ta",
    "Marathi Movies": "mr",
    "Malayalam Movies": "ma",
    "Kannada Movies": "kn"
}


# The on_change callback will be triggered whenever a new option is selected
def handle_language_selection():
    if st.session_state.language_choice != "All Languages":
        st.session_state.show_popular_movies_by_language = True
        st.session_state.show_results = False
    else:
        st.session_state.show_popular_movies_by_language = False


choice = st.sidebar.selectbox("Select a language:", list(language_options.keys()), key='language_choice',
                              on_change=handle_language_selection)

# ------------------ Conditional Display of Sections ------------------ #

# Display language-specific movies if a language is selected
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
        st.warning(f"No movies found for {st.session_state.language_choice}.")

# Display search results if a search was performed and a match was found
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
                else:
                    st.error(f"Could not fetch details for {title}.")

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

# Default view: show consolidated popular movies
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