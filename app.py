"""Gradio web app: a Pinterest-style wall of book recommendations."""
import html
from urllib.parse import quote

import gradio as gr

from recommender import (COVER_NOT_FOUND, TONE_COLUMNS, format_authors, popular_books,
                         retrieve_semantic_recommendations)

PLACEHOLDER_URL = f"/gradio_api/file={quote(str(COVER_NOT_FOUND))}"

# Free Hugging Face hosting runs on ZeroGPU, which requires at least one @spaces.GPU function at startup.
# Search runs on CPU plus the OpenAI API, so this function is never called. `spaces` only exists on Spaces.
try:
    import spaces

    @spaces.GPU
    def _zerogpu_placeholder():
        pass
except ImportError:
    pass


def book_card(row) -> str:
    cover = row["large_thumbnail"] if str(row["large_thumbnail"]).startswith("http") else PLACEHOLDER_URL
    words = row["description"].split()
    blurb = " ".join(words[:45]) + ("…" if len(words) > 45 else "")
    title = html.escape(row["title"])
    return f"""
    <article class="card">
      <div class="cover">
        <img src="{html.escape(cover)}" alt="Cover of {title}" loading="lazy">
        <div class="blurb"><p>{html.escape(blurb)}</p></div>
      </div>
      <div class="meta">
        <h3>{title}</h3>
        <p class="author">{html.escape(format_authors(row["authors"]))}</p>
        <div class="tags">
          <span class="tag">{html.escape(row["simple_categories"])}</span>
          <span class="tag">★ {row["average_rating"]:.1f}</span>
        </div>
      </div>
    </article>"""


def render_bookwall(query: str, category: str, tone: str) -> str:
    query = (query or "").strip()
    if query:
        try:
            recs = retrieve_semantic_recommendations(query, category, tone, final_top_k=24)
        except Exception:
            return '<div class="notice">Something went wrong while searching. Please try again in a moment.</div>'
        heading = f"Picked for “{html.escape(query)}”"
    else:
        recs = popular_books if category == "All" else popular_books[popular_books["simple_categories"] == category]
        recs = recs.head(24)
        if tone in TONE_COLUMNS:
            recs = recs.sort_values(TONE_COLUMNS[tone], ascending=False)
        heading = "Popular picks"

    if recs.empty:
        return '<div class="notice">No books found. Try another description or category.</div>'
    cards = "".join(book_card(row) for _, row in recs.iterrows())
    return f'<h2 class="wall-heading">{heading}</h2><div class="bookwall">{cards}</div>'


# (label shown, value used for filtering)
categories = [("All", "All"), ("Fiction", "Fiction"), ("Nonfiction", "Nonfiction"),
              ("Kids' Fiction", "Children's Fiction"), ("Kids' Nonfiction", "Children's Nonfiction")]
tones = ["All"] + list(TONE_COLUMNS)
suggestions = [
    "A story about forgiveness",
    "A cozy mystery in a small village",
    "Science fiction about first contact with aliens",
    "A book to teach children about nature",
    "Something uplifting after a hard week",
]

theme = gr.themes.Base(
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill="#faf7f2",
    body_background_fill_dark="#171513",
    block_background_fill="transparent",
    block_background_fill_dark="transparent",
    block_border_width="0px",
    button_primary_background_fill="#c8553d",
    button_primary_background_fill_hover="#b04832",
    button_primary_background_fill_dark="#e07a5f",
    button_primary_background_fill_hover_dark="#ea8f77",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#1a1714",
)

CSS = """
.gradio-container {
  --ink: #1f1b16; --muted: #6b645b; --line: #ebe5dc; --surface: #ffffff;
  --chip: #f1ece4; --accent: #c8553d; --accent-ink: #ffffff;
  max-width: 1280px !important; margin: 0 auto !important;
}
.dark .gradio-container {
  --ink: #f3eee7; --muted: #a9a197; --line: #34302b; --surface: #221f1c;
  --chip: #2c2824; --accent: #e07a5f; --accent-ink: #1a1714;
}
footer { display: none !important; }

/* Header */
#hero { text-align: center; padding: 36px 16px 8px; }
#hero h1 {
  font-family: 'Fraunces', Georgia, serif; font-weight: 600; font-size: clamp(2.2rem, 5vw, 3.4rem);
  letter-spacing: -0.02em; color: var(--ink); margin: 0;
}
#hero p { color: var(--muted); font-size: 1.05rem; margin: 10px auto 0; max-width: 560px; }

/* Search bar */
#searchbar { max-width: 760px; margin: 18px auto 0 !important; gap: 10px !important; align-items: center; }
#query textarea, #query input {
  border-radius: 999px !important; padding: 14px 22px !important; font-size: 1.05rem !important;
  background: var(--surface) !important; border: 1px solid var(--line) !important; color: var(--ink) !important;
  box-shadow: 0 2px 12px rgba(31, 27, 22, 0.06) !important;
}
#search-btn { border-radius: 999px !important; min-height: 50px; font-weight: 600; }

/* Suggestion and filter chips */
#suggestions { justify-content: center; gap: 8px !important; flex-wrap: wrap !important; margin-top: 4px; }
#suggestions button {
  flex: 0 0 auto !important; min-width: 0 !important; border-radius: 999px !important;
  background: transparent !important; border: 1px solid var(--line) !important;
  color: var(--muted) !important; font-weight: 500 !important; box-shadow: none !important;
}
#suggestions button:hover { color: var(--ink) !important; border-color: var(--muted) !important; }
#filters { justify-content: center; gap: 28px !important; margin-top: 6px; flex-wrap: wrap !important; }
#filters, #filters > *, #filters .form, #filters fieldset {
  background: transparent !important; border: none !important; box-shadow: none !important;
}
#filters > div { flex: 0 1 auto !important; min-width: 0 !important; }
.chips .wrap { gap: 6px !important; justify-content: center; }
.chips label {
  border-radius: 999px !important; padding: 6px 14px !important; background: var(--chip) !important;
  border: 1px solid transparent !important; color: var(--ink) !important; cursor: pointer; box-shadow: none !important;
}
.chips label:has(input:checked) { background: var(--ink) !important; color: var(--surface) !important; }
.chips input { display: none !important; }
.chips span[data-testid="block-info"] {
  color: var(--muted) !important; font-size: 0.8rem !important; text-transform: uppercase;
  letter-spacing: 0.08em; text-align: center; display: block; background: none !important;
}

/* Masonry wall */
.wall-heading {
  font-family: 'Fraunces', Georgia, serif; font-weight: 600; font-size: 1.5rem;
  color: var(--ink); margin: 26px 4px 16px;
}
.bookwall { column-count: 5; column-gap: 18px; }
@media (max-width: 1100px) { .bookwall { column-count: 4; } }
@media (max-width: 820px) { .bookwall { column-count: 3; } }
@media (max-width: 560px) { .bookwall { column-count: 2; column-gap: 12px; } }
.card { break-inside: avoid; display: inline-block; width: 100%; margin: 0 0 22px; }
.cover {
  position: relative; border-radius: 16px; overflow: hidden; background: var(--chip);
  outline: 1px solid var(--line); outline-offset: -1px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06), 0 6px 18px rgba(0, 0, 0, 0.08);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.cover img { display: block; width: 100%; height: auto; }
.blurb {
  position: absolute; inset: 0; display: flex; align-items: flex-end; padding: 14px;
  background: linear-gradient(to top, rgba(15, 12, 10, 0.92) 30%, rgba(15, 12, 10, 0.25));
  opacity: 0; transition: opacity 0.2s ease;
}
.blurb p { color: #f7f3ee; font-size: 0.82rem; line-height: 1.45; margin: 0; }
.card:hover .cover { transform: translateY(-3px); box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08), 0 14px 30px rgba(0, 0, 0, 0.14); }
.card:hover .blurb { opacity: 1; }
@media (hover: none) { .blurb { display: none; } }
.meta { padding: 10px 4px 0; }
.meta h3 {
  font-size: 0.95rem; font-weight: 600; line-height: 1.3; color: var(--ink); margin: 0;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}
.meta .author { color: var(--muted); font-size: 0.83rem; margin: 3px 0 0; }
.tags { display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; }
.tag { background: var(--chip); color: var(--muted); font-size: 0.72rem; padding: 3px 9px; border-radius: 999px; }
.notice { text-align: center; color: var(--muted); padding: 48px 16px; font-size: 1.05rem; }

/* Phones: stack the search bar and filters so nothing overflows sideways */
#searchbar > *, #filters > * { min-width: 0 !important; }
@media (max-width: 640px) {
  #hero { padding-top: 20px; }
  #searchbar { flex-direction: column !important; align-items: stretch !important; }
  #searchbar > * { width: 100% !important; }
  #filters { flex-direction: column !important; gap: 10px !important; }
  #suggestions button { font-size: 0.75rem !important; }
}
"""

HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&display=swap" rel="stylesheet">
"""

with gr.Blocks(title="Find your next read") as demo:
    gr.HTML("""
      <div id="hero">
        <h1>Find your next read</h1>
        <p>Describe the book you're in the mood for, and get recommendations matched by meaning, not keywords.</p>
      </div>""")

    with gr.Row(elem_id="searchbar"):
        user_query = gr.Textbox(show_label=False, placeholder="e.g. A story about forgiveness",
                                elem_id="query", scale=5, container=False)
        search_button = gr.Button("Search", variant="primary", elem_id="search-btn", scale=1, min_width=110)

    with gr.Row(elem_id="suggestions"):
        suggestion_buttons = [gr.Button(text, size="sm") for text in suggestions]

    with gr.Row(elem_id="filters"):
        category_choice = gr.Radio(categories, value="All", label="Category", elem_classes="chips")
        tone_choice = gr.Radio(tones, value="All", label="Mood", elem_classes="chips")

    bookwall = gr.HTML()

    inputs = [user_query, category_choice, tone_choice]
    for trigger in (search_button.click, user_query.submit, category_choice.change, tone_choice.change):
        trigger(render_bookwall, inputs, bookwall)
    for button, text in zip(suggestion_buttons, suggestions):
        button.click(lambda text=text: text, None, user_query).then(render_bookwall, inputs, bookwall)
    demo.load(render_bookwall, inputs, bookwall)


LAUNCH_OPTIONS = dict(theme=theme, css=CSS, head=HEAD, allowed_paths=[str(COVER_NOT_FOUND)])

if __name__ == "__main__":
    demo.launch(**LAUNCH_OPTIONS)