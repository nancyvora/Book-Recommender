"""Data loading and semantic search, shared by the web app (app.py), the API (api.py) and the tests."""
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
COVER_NOT_FOUND = BASE_DIR / "cover-not-found.jpg"

TONE_COLUMNS = {"Happy": "joy", "Surprising": "surprise", "Angry": "anger", "Suspenseful": "fear", "Sad": "sadness"}
CATEGORIES = ["Fiction", "Nonfiction", "Children's Fiction", "Children's Nonfiction"]

load_dotenv(BASE_DIR / ".env", override=True)

books = pd.read_csv(DATA_DIR / "processed" / "books_with_emotions.csv")
# https, so covers still load when the app is served over https
books["large_thumbnail"] = books["thumbnail"].str.replace("http://", "https://", regex=False) + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    str(COVER_NOT_FOUND),
    books["large_thumbnail"],
)

# Shown before the first search: well-loved books, no API call needed
popular_books = (
    books[(books["average_rating"] >= 3.8) & books["thumbnail"].notna()]
    .sort_values("ratings_count", ascending=False)
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db_books = Chroma(
    persist_directory=str(DATA_DIR / "chroma_db"),
    embedding_function=embeddings
)


def retrieve_semantic_recommendations(
        query: str,
        category: str = "All",
        tone: str = "All",
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    query_vector = embeddings.embed_query(query)
    k = initial_top_k
    while True:
        recs = db_books.similarity_search_by_vector(query_vector, k=k)
        books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
        # Keep Chroma's relevance order; filtering with isin() alone would return books in CSV order
        rank = {isbn: position for position, isbn in enumerate(books_list)}
        book_recs = books[books["isbn13"].isin(rank)].sort_values("isbn13", key=lambda s: s.map(rank))
        if category != "All":
            book_recs = book_recs[book_recs["simple_categories"] == category]
        # Small categories rarely reach the top 50, so search deeper until there are enough books
        if len(book_recs) >= final_top_k or k >= len(books):
            break
        k = min(k * 4, len(books))
    book_recs = book_recs.head(final_top_k)

    if tone in TONE_COLUMNS:
        book_recs = book_recs.sort_values(by=TONE_COLUMNS[tone], ascending=False)

    return book_recs


def format_authors(authors: str) -> str:
    names = str(authors).split(";")
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    if len(names) > 2:
        return f"{', '.join(names[:-1])}, and {names[-1]}"
    return names[0]
