import gradio as gr
import pandas as pd

from src.preprocessing import load_books
from src.recommender import BookRecommender

CSV_PATH = "data/books.csv"

# Load data and build model once (startup)
df = load_books(CSV_PATH)
recommender = BookRecommender.build(df)

# UX: list of titles (dropdown)
titles = sorted(list({t for t in df["display_title"].fillna("").astype(str).tolist() if t.strip()}))

# Dataset columns can vary
THUMB_COLS = ["thumbnail", "thumbnail_url", "image_url", "img_url", "cover"]
RATING_COLS = ["average_rating", "rating", "ratings"]
DESC_COLS = ["description", "desc", "summary"]


def pick_first(row, cols, default=""):
    for c in cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return row[c]
    return default


def short(text, n=260):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    return text if len(text) <= n else text[:n].rstrip() + "..."


def build_cards(results):
    if not results:
        return "<div style='opacity:.85; color:#e6edf3;'>No recommendations found. Try another input.</div>"

    cards = ""
    for r in results:
        title = r.get("display_title", "")
        authors = r.get("authors", "")
        categories = r.get("categories", "")
        rating = pick_first(r, RATING_COLS, "")
        desc = pick_first(r, DESC_COLS, "")
        thumb = pick_first(r, THUMB_COLS, "")
        sim = r.get("similarity", 0.0)

        img_html = ""
        if thumb:
            img_html = f"""
            <div style="flex:0 0 92px;">
              <img class="cover" src="{thumb}" />
            </div>
            """

        meta = []
        if authors:
            meta.append(f"<b>Authors:</b> {authors}")
        if categories:
            meta.append(f"<b>Categories:</b> {categories}")
        if rating != "":
            meta.append(f"<b>Rating:</b> {rating}")
        meta_html = "<br/>".join(meta)

        cards += f"""
        <div class="card" style="display:flex; gap:14px;">
          {img_html}
          <div style="flex:1;">
            <div style="display:flex; align-items:center; justify-content:space-between; gap:10px;">
              <div class="card-title">{title}</div>
              <div class="badge">Similarity: <b>{sim:.3f}</b></div>
            </div>
            <div class="card-meta">{meta_html}</div>
            <div class="card-desc">{short(desc)}</div>
          </div>
        </div>
        """
    return cards


def recommend_by_title_ui(title_query, top_n):
    res = recommender.recommend_by_title(title_query, top_n=int(top_n))
    return build_cards(res)


def recommend_by_text_ui(text_query, top_n):
    res = recommender.recommend_by_text(text_query, top_n=int(top_n))
    return build_cards(res)


def recommend_by_library_ui(liked_titles, top_n):
    res = recommender.recommend_by_library(liked_titles, top_n=int(top_n))
    return build_cards(res)


CSS = """
.gradio-container { max-width: 980px !important; }

body, .gradio-container {
  background: #0b1220 !important;
}

h1, h2, h3, p, label, span, .prose {
  color: #e6edf3 !important;
}

.card {
  background: #111a2e;
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 18px;
  padding: 14px;
  margin: 12px 0;
  box-shadow: 0 14px 34px rgba(0,0,0,.35);
}

.card-title { font-size: 18px; font-weight: 800; color: #ffffff; }
.card-meta { margin-top: 8px; font-size: 13px; color: rgba(230,237,243,.85); line-height: 1.45; }
.card-desc { margin-top: 10px; font-size: 13px; color: rgba(230,237,243,.78); line-height: 1.55; }

.badge {
  font-size: 12px;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(255,255,255,.10);
  color: #e6edf3;
}

img.cover {
  width: 92px;
  height: 128px;
  object-fit: cover;
  border-radius: 14px;
  box-shadow: 0 10px 24px rgba(0,0,0,.45);
}
"""


with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="Book Recommender") as demo:
    gr.Markdown(
        """
# 📚 Book Recommender System
Explainable recommendations using **TF-IDF + Cosine Similarity** over title/authors/categories/description.
"""
    )

    with gr.Tabs():
        with gr.Tab("Recommend by Title"):
            gr.Markdown("Choose a title (or type one) and get similar books.")
            with gr.Row():
                title_input = gr.Dropdown(
                    choices=titles,
                    label="Book title",
                    allow_custom_value=True,
                )
                topn1 = gr.Slider(1, 12, value=6, step=1, label="Top N")
            btn1 = gr.Button("Recommend", variant="primary")
            out1 = gr.HTML()
            btn1.click(recommend_by_title_ui, inputs=[title_input, topn1], outputs=out1)

        with gr.Tab("Recommend by Topic"):
            gr.Markdown("Write something like: **books about stoicism and habits**")
            with gr.Row():
                text_input = gr.Textbox(label="Topic / free text", placeholder="books about ...", lines=2)
                topn2 = gr.Slider(1, 12, value=6, step=1, label="Top N")
            btn2 = gr.Button("Recommend", variant="primary")
            out2 = gr.HTML()
            btn2.click(recommend_by_text_ui, inputs=[text_input, topn2], outputs=out2)

        with gr.Tab("My Library"):
            gr.Markdown("Pick a few books you like. We average their TF-IDF vectors to create a user profile.")
            liked = gr.Dropdown(
                choices=titles,
                multiselect=True,
                label="Select liked books",
            )
            topn3 = gr.Slider(1, 12, value=8, step=1, label="Top N")
            btn3 = gr.Button("Recommend based on my library", variant="primary")
            out3 = gr.HTML()
            btn3.click(recommend_by_library_ui, inputs=[liked, topn3], outputs=out3)

demo.launch()