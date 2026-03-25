"""
app.py – BookMuse
=================
Run with: python app.py
"""

import html
import json
import os

import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

print("KEY LOADED:", bool(os.getenv("OPENAI_API_KEY")))

from src.preprocessing import load_books
from src.recommender import BookRecommender
from src.agent_tools import (
    TOOL_SCHEMAS,
    filter_books,
    get_book_info,
    recommend_books,
    search_books,
)

# ── Startup ───────────────────────────────────────────────────────────────────
df = load_books("data/books.csv")
recommender = BookRecommender.build(df)

title_col = "display_title" if "display_title" in df.columns else "title"
titles = sorted({t for t in df[title_col].fillna("").astype(str) if t.strip()})

RATING_COLS = ["average_rating", "rating", "ratings"]
DESC_COLS = ["description", "desc", "summary"]
THUMB_COLS = ["thumbnail", "thumbnail_url", "image_url", "img_url", "cover"]

_genres = (
    df["categories"].fillna("").astype(str).str.split(r"[,;/]")
    .explode().str.strip().str.title()
)
GENRES = ["Any"] + sorted({g for g in _genres if len(g) > 2})

CURRENT_YEAR = 2024
OPENAI_TOOLS = [{"type": "function", "function": s} for s in TOOL_SCHEMAS]

MODES = [
    "Filter by genre / rating / year",
    "Search by mood / theme",
    "Books similar to a title",
    "Look up a specific book",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def pick(row, cols, default=""):
    for c in cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return row[c]
    return default


def clip(text, n=260):
    if not isinstance(text, str):
        text = str(text) if text else ""
    return text if len(text) <= n else text[:n].rstrip() + "…"


def esc(value):
    return html.escape("" if value is None else str(value))


def render_cards(results):
    """Render a list of books as HTML cards."""
    if not results:
        return "<div class='empty'>No results found — try a different search.</div>"

    html_out = ""
    for r in results:
        title = r.get("display_title", r.get("title", ""))
        author = r.get("authors", "")
        cats = r.get("categories", "")
        rating = r.get("rating", pick(r, RATING_COLS, ""))
        desc = r.get("description", pick(r, DESC_COLS, ""))
        thumb = r.get("thumbnail", pick(r, THUMB_COLS, ""))
        year = r.get("year", r.get("published_year", ""))

        safe_title = esc(title)
        safe_author = esc(author)
        safe_cats = esc(cats)
        safe_rating = esc(rating)
        safe_desc = esc(clip(desc, 220))
        safe_year = esc(year)
        safe_thumb = esc(thumb)

        cover = (
            f"<img class='cover' src='{safe_thumb}' "
            f"onerror=\"this.outerHTML='<div class=cover-ph>📖</div>'\">"
            if safe_thumb else "<div class='cover-ph'>📖</div>"
        )

        chips = ""
        if safe_author:
            chips += f"<span class='chip'>✍ {safe_author}</span>"
        if safe_cats:
            chips += f"<span class='chip'>🏷 {safe_cats}</span>"
        if safe_rating and safe_rating not in ["N/A", "nan"]:
            chips += f"<span class='chip chip-star'>★ {safe_rating}</span>"
        if safe_year:
            chips += f"<span class='chip'>📅 {safe_year}</span>"

        html_out += f"""
        <div class='card'>
          <div class='card-img'>{cover}</div>
          <div class='card-body'>
            <div class='card-top'>
              <span class='card-title'>{safe_title}</span>
            </div>
            <div class='card-chips'>{chips}</div>
            <div class='card-desc'>{safe_desc}</div>
          </div>
        </div>
        """

    return html_out


def render_book_detail(book):
    """Render one specific book as a detailed card."""
    if not isinstance(book, dict):
        return "<div class='empty'>No book data found.</div>"

    if book.get("error"):
        return f"<div class='empty'>{esc(book['error'])}</div>"

    title = esc(book.get("title", ""))
    author = esc(book.get("authors", ""))
    cats = esc(book.get("categories", ""))
    rating = esc(book.get("rating", ""))
    desc = esc(book.get("description", "No description available."))
    thumb = esc(book.get("thumbnail", ""))
    year = esc(book.get("year", ""))
    pages = esc(book.get("pages", ""))

    cover = (
        f"<img class='cover detail-cover' src='{thumb}' "
        f"onerror=\"this.outerHTML='<div class=cover-ph>📖</div>'\">"
        if thumb else "<div class='cover-ph detail-cover'>📖</div>"
    )

    chips = ""
    if author:
        chips += f"<span class='chip'>✍ {author}</span>"
    if cats:
        chips += f"<span class='chip'>🏷 {cats}</span>"
    if rating and rating not in ["N/A", "nan"]:
        chips += f"<span class='chip chip-star'>★ {rating}</span>"
    if year:
        chips += f"<span class='chip'>📅 {year}</span>"
    if pages:
        chips += f"<span class='chip'>📄 {pages} pages</span>"

    return f"""
    <div class='card detail-card'>
      <div class='card-img'>{cover}</div>
      <div class='card-body'>
        <div class='card-top'>
          <span class='card-title'>{title}</span>
        </div>
        <div class='card-chips'>{chips}</div>
        <div class='card-desc'>{desc}</div>
      </div>
    </div>
    """


# ── Ask AI ────────────────────────────────────────────────────────────────────
def ask_ai(question):
    question = (question or "").strip()
    if not question:
        return "Please enter a question."

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return "⚠ OPENAI_API_KEY not set in .env"

    books = recommender.recommend_by_text(question, top_n=12)

    context_lines = []
    for i, b in enumerate(books, 1):
        line = f"{i}. {b.get(title_col, b.get('title', ''))} by {b.get('authors', '')}"
        desc = pick(b, DESC_COLS, "")
        rating = pick(b, RATING_COLS, "")
        year = b.get("published_year", "")

        meta = []
        if rating:
            meta.append(f"rating: {rating}")
        if year:
            meta.append(f"year: {year}")
        if meta:
            line += f" ({', '.join(meta)})"
        if desc:
            line += f" — {clip(desc, 180)}"

        context_lines.append(line)

    context = "\n".join(context_lines) or "No books found."

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=900,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are BookMuse, a knowledgeable book advisor. "
                    "Answer using ONLY the books in the provided context. "
                    "Do not invent books, ratings, authors, or facts not present in the context. "
                    "Be helpful, clear, and concise."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nBooks in database:\n{context}",
            },
        ],
    )
    return resp.choices[0].message.content or "No answer generated."


# ── Agent ─────────────────────────────────────────────────────────────────────
def build_message(mode, genre, min_rating, year_from, year_to, mood, sim_title, lookup_title):
    if mode == "Filter by genre / rating / year":
        parts = ["Find me"]
        if genre and genre != "Any":
            parts.append(genre)
        parts.append("books")
        if min_rating > 0:
            parts.append(f"rated at least {min_rating:.1f}")
        if year_from > 1800 or year_to < CURRENT_YEAR:
            parts.append(f"published between {year_from} and {year_to}")
        return " ".join(parts) + "."

    if mode == "Search by mood / theme":
        q = (mood or "").strip() or "interesting books"
        return f"Find books about: {q}."

    if mode == "Books similar to a title":
        t = (sim_title or "").strip() or "a popular book"
        return f"Recommend books similar to '{t}'."

    if mode == "Look up a specific book":
        t = (lookup_title or "").strip() or "an interesting book"
        return f"Tell me about the book '{t}'."

    return "Show me some great book recommendations."


def _run_tool(name: str, args: dict) -> str:
    try:
        if name == "search_books":
            result = search_books(
                args.get("query", ""),
                df,
                recommender,
                args.get("top_n", 8),
            )
        elif name == "filter_books":
            result = filter_books(
                df,
                genre=args.get("genre", ""),
                min_rating=args.get("min_rating", 0.0),
                max_rating=args.get("max_rating", 5.0),
                year_from=args.get("year_from", 0),
                year_to=args.get("year_to", 9999),
                author=args.get("author", ""),
                top_n=args.get("top_n", 8),
            )
        elif name == "recommend_books":
            result = recommend_books(
                args.get("title", ""),
                recommender,
                args.get("top_n", 6),
            )
        elif name == "get_book_info":
            result = get_book_info(args.get("title", ""), df)
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as e:
        result = {"error": str(e)}

    return json.dumps(result, ensure_ascii=False)


def run_agent(mode, genre, min_rating, year_from, year_to, mood, sim_title, lookup_title):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return "<div class='empty'>⚠ OPENAI_API_KEY not set in .env</div>"

    user_msg = build_message(mode, genre, min_rating, year_from, year_to, mood, sim_title, lookup_title)
    client = OpenAI(api_key=api_key)

    messages = [
        {
            "role": "system",
            "content": (
                "You are BookMuse, a book recommendation assistant. "
                "You must always choose the best tool for the user's request. "
                "If the request is about a mood, theme, topic, genre, or vibe, use search_books. "
                "If the request asks for books similar to a known title, use recommend_books. "
                "If the request asks for detailed information about one book, use get_book_info. "
                "If the request is explicitly about year, rating, genre, or author filters, use filter_books. "
                "Do not answer from memory. Use tools."
            ),
        },
        {"role": "user", "content": user_msg},
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=OPENAI_TOOLS,
        tool_choice="auto",
        max_tokens=600,
    )

    message = response.choices[0].message
    tool_calls = getattr(message, "tool_calls", None)

    if not tool_calls:
        txt = esc(message.content or "No response generated.")
        return f"<div class='empty'>{txt}</div>"

    call = tool_calls[0]

    try:
        args = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}

    output = _run_tool(call.function.name, args)

    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return f"<div class='empty'>{esc(output)}</div>"

    title_html = f"<div class='agent-head'>{esc(user_msg)}</div>"

    if call.function.name == "get_book_info":
        return title_html + render_book_detail(data)

    if isinstance(data, list):
        return title_html + render_cards(data)

    if isinstance(data, dict) and data.get("error"):
        return f"<div class='empty'>{esc(data['error'])}</div>"

    return "<div class='empty'>No results found.</div>"


def clear_agent():
    return ""


def switch_panels(mode):
    return [
        gr.update(visible=(mode == MODES[0])),
        gr.update(visible=(mode == MODES[1])),
        gr.update(visible=(mode == MODES[2])),
        gr.update(visible=(mode == MODES[3])),
    ]


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Inter:wght@400;500&display=swap');

:root {
  --bg: #0f1117;
  --surface: #1a1d27;
  --surface2: #22263a;
  --border: #2e3250;
  --text: #e8eaf6;
  --muted: #8b90b8;
  --accent: #7c6af7;
  --accent2: #9d8ff9;
  --acbg: rgba(124,106,247,0.12);
  --shadow: 0 4px 24px rgba(0,0,0,0.4);
}

body, .gradio-container {
  background: var(--bg) !important;
  font-family: 'Inter', sans-serif !important;
  color: var(--text) !important;
}

.gradio-container {
  max-width: 1060px !important;
  width: 96% !important;
  margin: 0 auto !important;
}

.bm-header {
  text-align: center;
  padding: 40px 0 28px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 4px;
}

.bm-title {
  font-family: 'Sora', sans-serif;
  font-size: 34px;
  font-weight: 700;
  color: var(--text);
}

.bm-title span { color: var(--accent); }

.bm-sub {
  font-size: 14px;
  color: var(--muted);
  margin-top: 6px;
}

.tab-desc {
  font-size: 13px;
  color: var(--muted);
  padding: 10px 0 18px;
  line-height: 1.6;
  border-bottom: 1px solid var(--border);
  margin-bottom: 18px;
}

.mode-radio .wrap {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 8px !important;
}

.mode-radio label {
  background: var(--surface) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 999px !important;
  padding: 7px 18px !important;
  text-transform: none !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  color: var(--text) !important;
}

.card {
  display: flex;
  gap: 16px;
  padding: 16px;
  margin: 12px 0;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  box-shadow: var(--shadow);
  transition: transform .18s, border-color .18s;
}

.card:hover {
  transform: translateY(-2px);
  border-color: var(--accent);
}

.card-img {
  flex: 0 0 84px;
}

.cover {
  width: 84px;
  height: 120px;
  object-fit: cover;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0,0,0,.45);
  display: block;
}

.detail-cover {
  width: 110px;
  height: 155px;
}

.cover-ph {
  width: 84px;
  height: 120px;
  background: var(--surface2);
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 28px;
  color: var(--muted);
  border: 1px solid var(--border);
}

.detail-cover.cover-ph {
  width: 110px;
  height: 155px;
}

.card-body {
  flex: 1;
  min-width: 0;
}

.card-top {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 8px;
}

.card-title {
  font-family: 'Sora', sans-serif;
  font-size: 17px;
  font-weight: 600;
  color: var(--text);
  line-height: 1.3;
}

.card-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 10px;
}

.chip {
  font-size: 12px;
  color: var(--muted);
  background: var(--surface2);
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
}

.chip-star {
  color: #fbbf24;
  background: rgba(251,191,36,.08);
  border-color: rgba(251,191,36,.2);
  font-weight: 600;
}

.card-desc {
  font-size: 13px;
  color: #c6cae3;
  line-height: 1.65;
}

.detail-card {
  padding: 20px;
}

.empty {
  text-align: center;
  padding: 42px 20px;
  color: var(--muted);
  font-size: 15px;
  background: var(--surface);
  border-radius: 12px;
  border: 1px dashed var(--border);
  margin-top: 12px;
}

.agent-head {
  font-size: 13px;
  color: var(--muted);
  margin: 6px 0 14px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
}
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="BookMuse", css=CSS) as demo:
    gr.HTML("""
    <div class="bm-header">
      <div class="bm-title">Book<span>Muse</span></div>
      <div class="bm-sub">Content-based book recommendations — TF-IDF · Cosine Similarity · GPT-4o</div>
    </div>
    """)

    with gr.Tabs():
        with gr.Tab("🤖 Ask AI"):
            gr.HTML(
                "<div class='tab-desc'>Ask anything in natural language. "
                "The app retrieves relevant books from your dataset and GPT answers using only those books.</div>"
            )

            ask_q = gr.Textbox(
                label="Your question",
                placeholder="e.g. What's a good philosophical novel for someone who liked Dostoevsky?",
                lines=3,
            )
            ask_btn = gr.Button("Ask", variant="primary")
            ask_out = gr.Textbox(label="Answer", lines=14, interactive=False)

            ask_btn.click(ask_ai, inputs=[ask_q], outputs=[ask_out])

        with gr.Tab("🔧 Agent"):
            gr.HTML(
                "<div class='tab-desc'>Choose what you want. "
                "The AI agent picks the right tool automatically and returns real results.</div>"
            )

            mode = gr.Radio(
                choices=MODES,
                value=MODES[0],
                label="What do you want to do?",
                elem_classes=["mode-radio"],
            )

            with gr.Group(visible=True) as p_filter:
                with gr.Row():
                    a_genre = gr.Dropdown(choices=GENRES, value="Any", label="Genre")
                    a_rating = gr.Slider(0.0, 5.0, value=0.0, step=0.1, label="Minimum rating (0 = any)")
                with gr.Row():
                    a_yr_from = gr.Slider(1800, CURRENT_YEAR, value=1900, step=1, label="Published from")
                    a_yr_to = gr.Slider(1800, CURRENT_YEAR, value=CURRENT_YEAR, step=1, label="Published until")

            with gr.Group(visible=False) as p_mood:
                a_mood = gr.Textbox(
                    label="Describe mood or theme",
                    placeholder="e.g. melancholic love story set in wartime…",
                    lines=2,
                )

            with gr.Group(visible=False) as p_similar:
                a_sim = gr.Dropdown(
                    choices=titles,
                    label="Book to base recommendations on",
                    allow_custom_value=True,
                )

            with gr.Group(visible=False) as p_lookup:
                a_lkp = gr.Dropdown(
                    choices=titles,
                    label="Book to look up",
                    allow_custom_value=True,
                )

            mode.change(
                switch_panels,
                inputs=[mode],
                outputs=[p_filter, p_mood, p_similar, p_lookup],
            )

            with gr.Row():
                a_run = gr.Button("Run Agent", variant="primary")
                a_clear = gr.Button("Clear", variant="secondary")

            agent_out = gr.HTML()

            a_run.click(
                run_agent,
                inputs=[mode, a_genre, a_rating, a_yr_from, a_yr_to, a_mood, a_sim, a_lkp],
                outputs=[agent_out],
            )

            a_clear.click(
                clear_agent,
                outputs=[agent_out],
            )

demo.launch()