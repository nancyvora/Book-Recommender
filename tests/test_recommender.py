import recommender
from recommender import books, format_authors, retrieve_semantic_recommendations

GILEAD = 9780002005883  # first book in the CSV


def last_book(category):
    return int(books.loc[books["simple_categories"] == category, "isbn13"].iloc[-1])


def test_results_follow_relevance_order_not_csv_order(query_like, stored_vectors):
    """Regression test: results used to come back in CSV file order instead of relevance order."""
    target = query_like(last_book("Nonfiction"))  # a book far down the CSV
    results = retrieve_semantic_recommendations("ignored", final_top_k=16)

    expected = [int(doc.page_content.strip('"').split()[0])
                for doc in recommender.db_books.similarity_search_by_vector(stored_vectors[target].tolist(), k=16)]
    assert results["isbn13"].tolist() == expected
    assert results["isbn13"].iloc[0] == target


def test_category_filter_keeps_relevance_order(query_like, stored_vectors):
    query_like(GILEAD)
    results = retrieve_semantic_recommendations("ignored", category="Fiction", final_top_k=16)

    assert (results["simple_categories"] == "Fiction").all()
    ranked = [int(doc.page_content.strip('"').split()[0])
              for doc in recommender.db_books.similarity_search_by_vector(stored_vectors[GILEAD].tolist(), k=200)]
    positions = [ranked.index(isbn) for isbn in results["isbn13"]]
    assert positions == sorted(positions)


def test_small_category_still_returns_a_full_page(query_like):
    """Regression test: Children's Nonfiction used to return as few as 1 book."""
    query_like(GILEAD)  # an adult novel, so few children's nonfiction books are nearby
    results = retrieve_semantic_recommendations("ignored", category="Children's Nonfiction", final_top_k=16)

    assert len(results) == 16
    assert (results["simple_categories"] == "Children's Nonfiction").all()


def test_tone_sorts_by_matching_emotion(query_like):
    query_like(GILEAD)
    for tone, column in recommender.TONE_COLUMNS.items():
        results = retrieve_semantic_recommendations("ignored", tone=tone)
        assert results[column].is_monotonic_decreasing, tone


def test_format_authors():
    assert format_authors("Jane Austen") == "Jane Austen"
    assert format_authors("A;B") == "A and B"
    assert format_authors("A;B;C") == "A, B, and C"
