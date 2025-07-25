# Book Recommendation System (LLM-powered)

## Overview
This project is an interactive Book Recommendation System that leverages semantic search and emotion-based filtering to suggest books tailored to a user's query, preferred genre, and desired emotional tone. It uses machine learning embeddings, a vector database (Chroma), and a Gradio web interface for a seamless user experience.

## Features
- **Semantic Search:** Finds books similar to a user's query using sentence-transformer embeddings.
- **Emotion Filtering:** Recommends books based on emotional tone (Happy, Sad, Angry, Surprising, Suspenseful).
- **Genre Selection:** Filter recommendations by book genre/category.
- **Interactive Dashboard:** User-friendly web interface built with Gradio.
- **Thumbnail Previews:** Displays book covers and short descriptions.

## Project Structure
- `app.py` — Main application file. Loads data, sets up the vector database, defines recommendation logic, and launches the Gradio dashboard.
- `books_with_emotions.csv` — Main dataset with book metadata, emotion scores, and thumbnails.
- `tagged_description.txt` — Text file with book descriptions for embedding.
- `chroma_db/` — Directory for the persistent Chroma vector database.
- `cover-not-found.png` — Placeholder image for missing book covers.
- `README.md` — Project documentation (this file).
- Other CSVs and notebooks — Data exploration and analysis.

## How It Works
1. **Data Loading:**
   - Loads book data from `books_with_emotions.csv`.
   - Prepares large thumbnails and handles missing covers.
2. **Vector Database:**
   - Uses Chroma to store and search book embeddings.
   - If the database exists, it loads it; otherwise, it creates a new one from `tagged_description.txt`.
3. **Recommendation Logic:**
   - Retrieves top semantic matches for the user's query.
   - Filters by genre and sorts by the selected emotional tone.
   - Returns book covers and short descriptions.
4. **Web Interface:**
   - Built with Gradio for easy interaction.
   - Users enter a query, select genre and tone, and receive recommendations in a gallery format.

## Setup & Usage
1. **Install Dependencies:**
   - Python 3.8+
   - Install required packages:
     ```bash
     pip install -r requirements.txt
     ```
2. **Prepare Data:**
   - Ensure `books_with_emotions.csv` and `tagged_description.txt` are present in the project directory.
3. **Run the App:**
   ```bash
   python app.py
   ```
   - The Gradio dashboard will launch in your browser.

## Customization
- **Add More Emotions:** Update the emotion columns in the CSV and adjust the sorting logic in `app.py`.
- **Change Embedding Model:** Modify the model name in the `HuggingFaceEmbeddings` instantiation.
- **UI Tweaks:** Edit the Gradio blocks in `app.py` for a different look or layout.


## How Each File is Used in This Project

- **app.py**: The main Python script that runs the entire application. It loads the book data, manages the vector database (Chroma), defines the recommendation logic, and builds the Gradio web interface for user interaction. All core logic and UI are here.

- **books_with_emotions.csv**: The primary dataset containing book metadata (title, authors, ISBN, etc.), emotion scores (joy, sadness, anger, surprise, fear), genres, and thumbnail URLs. This file is loaded by `app.py` to provide the data for recommendations.

- **tagged_description.txt**: Contains the text descriptions of books, one per line, tagged with identifiers. These descriptions are embedded using a sentence-transformer model and stored in the Chroma vector database for semantic search. Used only when initializing or rebuilding the vector database.

- **chroma_db/**: A directory that stores the persistent Chroma vector database files. This database holds the embeddings for book descriptions and enables fast semantic search. It is automatically created and managed by `app.py`.

- **cover-not-found.png**: A placeholder image used when a book's cover thumbnail is missing or unavailable. Ensures the UI always displays an image for each book.

- **README.md**: This documentation file. Explains the project, setup, usage, and describes the purpose of each file.

- **requirements.txt**: Lists all Python dependencies needed to run the project. Install these with `pip install -r requirements.txt`.

- **BooksDataset.csv, books.csv, cleaned_books.csv**: Additional or legacy datasets that may have been used for data exploration, cleaning, or development. Not directly used by the main app, but useful for reference or further analysis.

- **data-exploration.ipynb**: Jupyter notebook for exploring and understanding the book data. Useful for data analysis and visualization during development.

- **sentiment-analysis.ipynb**: Jupyter notebook for analyzing the sentiment or emotional content of book descriptions. Likely used to generate the emotion scores in the main dataset.

- **text-classification.ipynb**: Jupyter notebook for experimenting with text classification models, possibly for genre or emotion tagging.

- **vector-search.ipynb**: Jupyter notebook for testing and visualizing vector search techniques, embeddings, and similarity queries.

## Credits
- Built with [Gradio](https://gradio.app/), [LangChain](https://python.langchain.com/), [ChromaDB](https://www.trychroma.com/), and [HuggingFace Transformers](https://huggingface.co/).

## License
This project is for educational and research purposes. See `LICENSE` if present.
