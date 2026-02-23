import gradio as gr
import pandas as pd

from src.preprocessing import load_books
from src.recommender import BookRecommender


CSV_PATH = "data/books.csv"

# Load data and build recommender once at startup
df = load_books(CSV_PATH)
recommender = BookRecommender.build(df)


def format_results(results):
    """Return a nice markdown output for Gradio."""
    if not results:
        return "No recommendations found. Try another title (check spelling)."

    md = "## Recommendations\n"
    for r in results:
        title = r.get("display_title", "")
        authors = r.get("authors", "")
        categories = r.get("categories", "")
        rating = r.get("average_rating", r.get("ratings", ""))  # depends on dataset column name
        description = r.get("description", "")

        # Keep description short
        if isinstance(description, str) and len(description) > 350:
            description = description[:350] + "..."

        md += f"\n### {title}\n"
        if authors:
            md += f"**Authors:** {authors}\n\n"
        if categories:
            md += f"**Categories:** {categories}\n\n"
        if rating != "":
            md += f"**Rating:** {rating}\n\n"
        if description:
            md += f"{description}\n\n"
        md += f"**Similarity:** {r.get('similarity', 0):.3f}\n"

    return md


def recommend(title_query, top_n):
    results = recommender.recommend_by_title(title_query, top_n=int(top_n))
    return format_results(results)


with gr.Blocks(title="Book Recommender") as demo:
    gr.Markdown("# 📚 Book Recommender System")
    gr.Markdown(
        "Type a book title from the dataset and get similar recommendations using **TF-IDF + Cosine Similarity**."
    )

    with gr.Row():
        title_input = gr.Textbox(label="Enter a book title", placeholder="e.g., The Hobbit")
        topn_input = gr.Slider(1, 10, value=5, step=1, label="Number of recommendations")

    btn = gr.Button("Recommend")
    output = gr.Markdown()

    btn.click(fn=recommend, inputs=[title_input, topn_input], outputs=output)

demo.launch()