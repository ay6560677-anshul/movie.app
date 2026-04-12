import streamlit as st
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
)


st.markdown("""
<style>
    /* Background */
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: #f0f0f0; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: rgba(255,255,255,0.05); border-right: 1px solid rgba(255,255,255,0.1); }
    section[data-testid="stSidebar"] * { color: #f0f0f0 !important; }

    /* Hero title */
    .hero { text-align: center; padding: 2rem 0 1rem; }
    .hero h1 { font-size: 3rem; font-weight: 800; background: linear-gradient(90deg, #f5a623, #f76b1c); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero p  { color: #bbb; font-size: 1.1rem; }

    /* Movie card */
    .movie-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        transition: transform 0.2s;
    }
    .movie-card:hover { transform: translateY(-3px); background: rgba(255,255,255,0.12); }
    .rank-badge {
        display: inline-block;
        background: linear-gradient(90deg, #f5a623, #f76b1c);
        color: #000;
        font-weight: 800;
        border-radius: 50%;
        width: 30px; height: 30px;
        line-height: 30px;
        text-align: center;
        margin-right: 10px;
        font-size: 0.85rem;
    }
    .movie-title { font-size: 1.05rem; font-weight: 600; color: #fff; }
    .movie-meta  { font-size: 0.82rem; color: #aaa; margin-top: 4px; }

    /* Stat box */
    .stat-box {
        background: rgba(245,166,35,0.1);
        border: 1px solid rgba(245,166,35,0.3);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stat-number { font-size: 2rem; font-weight: 800; color: #f5a623; }
    .stat-label  { font-size: 0.8rem; color: #aaa; }

    /* Similarity bar */
    .sim-bar-bg { background: rgba(255,255,255,0.1); border-radius: 6px; height: 6px; margin-top: 6px; }
    .sim-bar-fg { background: linear-gradient(90deg, #f5a623, #f76b1c); height: 6px; border-radius: 6px; }

    /* Input overrides */
    .stTextInput > div > div > input { background: rgba(255,255,255,0.08) !important; color: #fff !important; border: 1px solid rgba(255,255,255,0.2) !important; border-radius: 10px !important; }
    div[data-testid="stSelectbox"] > div { background: rgba(255,255,255,0.08) !important; border: 1px solid rgba(255,255,255,0.2) !important; border-radius: 10px !important; }
    div[data-testid="stSelectbox"] * { color: #fff !important; }
</style>
""", unsafe_allow_html=True)



@st.cache_data(show_spinner="🎬 Loading movie database…")
def load_and_build(csv_path: str):
    movies = pd.read_csv(csv_path)
    movies.reset_index(drop=False, inplace=True)   # ensure 'index' column exists
    if 'index' not in movies.columns:
        movies['index'] = movies.index

    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feat in selected_features:
        if feat in movies.columns:
            movies[feat] = movies[feat].fillna('')

    existing = [f for f in selected_features if f in movies.columns]
    combined = movies[existing[0]]
    for f in existing[1:]:
        combined = combined + ' ' + movies[f]

    vectorizer = TfidfVectorizer()
    feature_vector = vectorizer.fit_transform(combined)
    similarity = cosine_similarity(feature_vector)
    return movies, similarity


def get_recommendations(movie_name: str, movies, similarity, top_n: int = 10):
    titles = movies['title'].tolist()
    close = difflib.get_close_matches(movie_name, titles, n=1, cutoff=0.3)
    if not close:
        return None, None, []
    matched = close[0]
    idx = movies[movies['title'] == matched]['index'].values[0]
    scores = list(enumerate(similarity[idx]))
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)

    results = []
    for movie_idx, score in scores_sorted[1: top_n + 1]:
        row = movies[movies['index'] == movie_idx]
        if row.empty:
            continue
        row = row.iloc[0]
        results.append({
            'title':    row.get('title', 'N/A'),
            'genres':   row.get('genres', ''),
            'director': row.get('director', ''),
            'cast':     row.get('cast', '')[:60] + '…' if len(str(row.get('cast', ''))) > 60 else row.get('cast', ''),
            'score':    round(float(score), 4),
        })
    return matched, idx, results



with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    csv_path = st.text_input("📂 CSV file path", value="movies.csv", help="Path to your movies.csv dataset")
    top_n    = st.slider("🎯 Number of recommendations", 5, 20, 10)

    st.markdown("---")
    st.markdown("### 📖 How it works")
    st.markdown("""
1. **TF-IDF Vectorizer** converts movie metadata (genres, keywords, tagline, cast, director) into feature vectors.
2. **Cosine Similarity** measures how close two movies are in vector space.
3. Top similar movies are returned ranked by similarity score.
    """)



st.markdown("""
<div class="hero">
  <h1>🎬 Movie Recommender</h1>
  <p>Content-based recommendations powered by TF-IDF &amp; Cosine Similarity</p>
</div>
""", unsafe_allow_html=True)


try:
    movies, similarity = load_and_build(csv_path)
except FileNotFoundError:
    st.error(f"❌ Could not find `{csv_path}`. Please update the path in the sidebar.")
    st.stop()


c1, c2, c3, c4 = st.columns(4)
for col, num, label in [
    (c1, f"{len(movies):,}", "Total Movies"),
    (c2, movies['genres'].nunique() if 'genres' in movies.columns else "—", "Unique Genre Tags"),
    (c3, movies['director'].nunique() if 'director' in movies.columns else "—", "Directors"),
    (c4, f"{similarity.shape[0]:,}²", "Similarity Matrix"),
]:
    col.markdown(f"""
    <div class="stat-box">
        <div class="stat-number">{num}</div>
        <div class="stat-label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")


st.markdown("### 🔍 Search a Movie")
col_input, col_btn = st.columns([4, 1])
with col_input:
    user_input = st.text_input("", placeholder="e.g. The Dark Knight, Inception, Avengers…", label_visibility="collapsed")
with col_btn:
    search_clicked = st.button("Recommend 🚀", use_container_width=True)


if search_clicked and user_input.strip():
    matched, idx, recs = get_recommendations(user_input.strip(), movies, similarity, top_n)

    if not recs:
        st.warning("⚠️ No close match found. Try a different movie title.")
    else:
        st.success(f"🎯 Matched **{matched}** — showing top {len(recs)} recommendations")

    
        left_col, right_col = st.columns(2)
        for i, rec in enumerate(recs):
            pct = int(rec['score'] * 100)
            card_html = f"""
            <div class="movie-card">
                <div><span class="rank-badge">{i+1}</span><span class="movie-title">{rec['title']}</span></div>
                <div class="movie-meta">🎭 {rec['genres'] or 'N/A'} &nbsp;|&nbsp; 🎬 {rec['director'] or 'N/A'}</div>
                <div class="movie-meta">⭐ Cast: {rec['cast'] or 'N/A'}</div>
                <div class="sim-bar-bg"><div class="sim-bar-fg" style="width:{pct}%"></div></div>
                <div class="movie-meta" style="text-align:right">Similarity: {rec['score']:.2%}</div>
            </div>"""
            if i % 2 == 0:
                left_col.markdown(card_html, unsafe_allow_html=True)
            else:
                right_col.markdown(card_html, unsafe_allow_html=True)

        
        st.markdown("### 📊 Similarity Score Comparison")
        chart_df = pd.DataFrame(recs)[['title', 'score']].set_index('title')
        st.bar_chart(chart_df)

elif search_clicked:
    st.warning("Please enter a movie name first.")


st.markdown("---")
with st.expander("📋 Explore Dataset"):
    show_cols = [c for c in ['title', 'genres', 'director', 'cast', 'keywords'] if c in movies.columns]
    st.dataframe(movies[show_cols].head(100), use_container_width=True, height=300)

 
st.markdown("<br><div style='text-align:center;color:#555;font-size:0.8rem'>Built with Streamlit · TF-IDF · Cosine Similarity</div>", unsafe_allow_html=True)
