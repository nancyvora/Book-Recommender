"""FastAPI service: a JSON recommendation endpoint, with the Gradio web app mounted at /.

Run locally with:  uvicorn api:app --port 7860
"""
from typing import Literal

import gradio as gr
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import LAUNCH_OPTIONS, demo
from recommender import CATEGORIES, TONE_COLUMNS, books, format_authors, retrieve_semantic_recommendations

Category = Literal["All", "Fiction", "Nonfiction", "Children's Fiction", "Children's Nonfiction"]
Tone = Literal["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
assert set(Category.__args__) == {"All", *CATEGORIES} and set(Tone.__args__) == {"All", *TONE_COLUMNS}


class RecommendationRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, examples=["A story about forgiveness"])
    category: Category = "All"
    tone: Tone = "All"
    top_k: int = Field(16, ge=1, le=50)


class Book(BaseModel):
    isbn13: int
    title: str
    authors: str
    category: str
    average_rating: float
    published_year: int | None
    description: str
    cover_url: str | None


class RecommendationResponse(BaseModel):
    query: str
    results: list[Book]


app = FastAPI(
    title="Semantic Book Recommender API",
    description="Recommends books from a natural-language description, using OpenAI embeddings and Chroma.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "books": len(books)}


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    try:
        recs = retrieve_semantic_recommendations(request.query, request.category, request.tone,
                                                 final_top_k=request.top_k)
    except Exception as error:
        raise HTTPException(status_code=503, detail="Search is temporarily unavailable.") from error
    results = [
        Book(
            isbn13=row["isbn13"],
            title=row["title"],
            authors=format_authors(row["authors"]),
            category=row["simple_categories"],
            average_rating=row["average_rating"],
            published_year=None if pd.isna(row["published_year"]) else int(row["published_year"]),
            description=row["description"],
            cover_url=row["large_thumbnail"] if str(row["large_thumbnail"]).startswith("http") else None,
        )
        for _, row in recs.iterrows()
    ]
    return RecommendationResponse(query=request.query, results=results)


# The web app is served at / (the API docs are at /docs)
app = gr.mount_gradio_app(app, demo, path="/", **LAUNCH_OPTIONS)
