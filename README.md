# 🎬 Movie Recommendation System  

A *Movie Recommendation Web App* built with [Streamlit](https://streamlit.io/).  
This app suggests movies based on user search and provides popular movies across different languages.  

It uses *TF-IDF + Nearest Neighbors* for recommendation, *fuzzy matching* for search, and the *OMDb API* for fetching posters and movie details.  

---

## 🚀 Features  

- 🔍 Search movies by title (fuzzy search supported).  
- 🎥 Handles multiple movies with the same/similar title.  
- ⭐ Movie Recommendations based on language-specific models.  
- 🌍 Browse popular movies by language (English, Hindi, Telugu, Tamil, Marathi, Malayalam, Kannada).  
- 📌 Posters, IMDb ratings, and plot summaries via OMDb API.  
- 🎨 Clean, responsive UI with hover effects and clickable IMDb links.  

---

## 🛠 Tech Stack  

- *Python 3.9+*  
- [Streamlit](https://streamlit.io/) – Web Framework  
- *Scikit-learn* – TF-IDF + Nearest Neighbors  
- *Pandas* – Data Handling  
- *OMDb API* – Movie Posters & Metadata  
- *Pickle* – Pre-trained model storage  

---

## 📂 Project Structure
📦 Movie-Recommender ┣ 📜 app.py                # Main Streamlit application ┣ 📜 movie_recommender.pkl # Pre-trained recommendation data (not included due to size) ┣ 📜 requirements.txt      # Python dependencies ┣ 📜 README.md             # Project documentation ┗ 📂 .streamlit            # Streamlit config (optional)

---

## ⚙ Installation  

1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/movie-recommender.git
   cd movie-recommender

2. Create and activate a virtual environment (recommended)

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows


3. Install dependencies

pip install -r requirements.txt




---

🔑 Setup

1. Get a free API key from OMDb API.


2. Replace the API key inside app.py:

api_key = "YOUR_API_KEY"


3. Place the pre-trained movie_recommender.pkl file in the project root directory.

(If the file is too large for GitHub, host it on Google Drive / Kaggle / HuggingFace and update the link here.)





---

▶ Run the App

streamlit run app.py

Then open in your browser at 👉 http://localhost:8501


---
## 🎯 Live Demo  
👉 [Movie Recommender on Render](https://movies-recommendation-system-0.onrender.com)
---

📝 Future Improvements

Add user login & personalized recommendations.

Deploy on Streamlit Cloud / HuggingFace Spaces / Heroku.

Improve recommendation algorithm (deep learning, embeddings).



---

📜 License

This project is licensed under the MIT License.

---
