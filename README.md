# 📚 Book Recommender System  
**NLP Project – EADA**

---

## 1. Project Motivation

The goal of this project is to build a content-based book recommender system using Natural Language Processing (NLP) techniques.

The system recommends books based on textual similarity, using information such as:

- Title  
- Authors  
- Categories  
- Description  

The objective is not only to build a working application, but to understand and explain the full NLP pipeline behind the recommendations.

---

## 2. Dataset Description

We use a CSV dataset containing book metadata, including:

- `title`
- `authors`
- `categories`
- `description`
- `average_rating`
- `ratings_count`
- `thumbnail`

For our recommendation logic, we focus mainly on textual fields:

- Title  
- Authors  
- Categories  
- Description  

These fields are combined into a single text representation per book.

---

## 3. Methodology

### 3.1 Preprocessing

The preprocessing pipeline includes:

1. Loading the dataset
2. Handling missing values
3. Combining textual fields into a single column (`full_text`)
4. Converting text to lowercase
5. Preparing text for vectorization

Each book is therefore represented as a unified textual description.

---

### 3.2 Text Representation – TF-IDF

We use **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert text into numerical vectors.

TF-IDF assigns higher weights to words that are:

- Frequent in a specific book  
- Rare across the entire dataset  

This helps highlight meaningful and distinctive words.

Each book becomes a numerical vector in a high-dimensional space.

---

### 3.3 Similarity Computation – Cosine Similarity

To measure similarity between books, we use **cosine similarity**.

Cosine similarity measures the angle between two vectors:

- Value close to 1 → very similar
- Value close to 0 → unrelated

When a user selects a book (or enters a topic), we:

1. Retrieve its vector
2. Compare it with all other book vectors
3. Rank them by similarity
4. Return the Top-K results

---

## 4. Recommendation Modes

### 1️⃣ Recommend by Title  
The user selects a book title.  
The system finds books with similar textual content.

### 2️⃣ Recommend by Topic  
The user enters free text (e.g., "books about stoicism and habits").  
The system converts the query into a TF-IDF vector and compares it with book vectors.

### 3️⃣ My Library  
The user selects multiple books they like.  
We average their TF-IDF vectors to create a simple user profile and recommend similar books.

---

## 5. System Architecture

User → Gradio Interface → BookRecommender → TF-IDF Vectors → Cosine Similarity → Ranked Results → UI Display

The system is modular:

- `src/preprocessing.py` → data cleaning and preparation  
- `src/recommender.py` → recommendation logic  
- `app.py` → Gradio user interface  

---

## 6. Example Usage

After launching the application, a local web interface will open in the browser.

Users can interact with the system in three ways:

1. **Find similar books**  
   Select a book title and the system will recommend similar books.

2. **Explore by topic**  
   Enter a description such as *"books about philosophy and ethics"* and the system will recommend relevant books.

3. **My Library**  
   Select multiple books you like and the system will generate recommendations based on their combined profile.

The recommendations are displayed with:
- book title
- authors
- categories
- rating
- similarity score
- short description

---

## 7. Limitations

This project uses a content-based recommendation approach, which has some limitations:

- It relies only on book metadata and textual descriptions.
- It does not use user behavior or collaborative filtering.
- TF-IDF captures word frequency but not deeper semantic meaning.
- Recommendations may be limited if book descriptions are very short or incomplete.

---

## 8. Future Improvements

Possible future extensions of the system include:

- Category-based browsing (searching books by genre)
- Using modern text embeddings such as Sentence-BERT
- Building a hybrid recommendation system (content + collaborative filtering)
- Adding user accounts and personalized recommendations
- Deploying the system as an online web application

---

## 9. How to Run the Project

Clone the repository and run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py

