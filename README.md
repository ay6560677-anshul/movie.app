# 🎬 Movie Recommender System

A **content-based movie recommendation system** built using Python, Machine Learning, and Streamlit. It recommends movies similar to the movie entered by the user using **TF-IDF** and **Cosine Similarity** based on movie features such as genres, keywords, tagline, cast, and director.

## ⚙️ Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install numpy pandas scikit-learn streamlit
```

## ▶️ How to Run

Make sure these files are in the same folder:

```text
app.py
movies.csv
```

Then run:

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

## 🧠 How It Works

1. Movie metadata is combined into text features.
2. **TF-IDF** converts the text into numerical vectors.
3. **Cosine Similarity** calculates similarity between movies.
4. The system displays the most similar movies as recommendations.

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* TF-IDF
* Cosine Similarity
