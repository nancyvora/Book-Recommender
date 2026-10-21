import pytest
from fastapi.testclient import TestClient

import recommender
from api import app
from app import render_bookwall

GILEAD = 9780002005883


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "books": len(recommender.books)}


def test_recommend_returns_books(client, query_like):
    query_like(GILEAD)
    response = client.post("/recommend", json={"query": "a quiet novel about faith", "top_k": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "a quiet novel about faith"
    assert len(body["results"]) == 5
    assert body["results"][0]["isbn13"] == GILEAD
    assert set(body["results"][0]) >= {"title", "authors", "category", "average_rating", "cover_url"}


@pytest.mark.parametrize("payload", [
    {"query": ""},
    {"query": "x", "category": "Poetry"},
    {"query": "x", "tone": "Bored"},
    {"query": "x", "top_k": 0},
    {"query": "x", "top_k": 51},
])
def test_recommend_rejects_invalid_input(client, payload):
    assert client.post("/recommend", json=payload).status_code == 422


def test_recommend_reports_search_outage(client, monkeypatch):
    def fail(self, text):
        raise ConnectionError("OpenAI unreachable")
    monkeypatch.setattr(type(recommender.embeddings), "embed_query", fail)

    response = client.post("/recommend", json={"query": "anything"})
    assert response.status_code == 503


def test_web_app_is_mounted(client):
    assert client.get("/").status_code == 200


def test_bookwall_shows_popular_books_without_a_search():
    page = render_bookwall("", "All", "All")
    assert "Popular picks" in page
    assert page.count('class="card"') == 24


def test_bookwall_escapes_the_query(query_like):
    query_like(GILEAD)
    page = render_bookwall('<script>alert("x")</script>', "All", "All")
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
