import pandas as pd
import numpy as np
from dotenv import load_dotenv


from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings
import os


load_dotenv()


books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"]  + "&fife=w800"

books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.png",
    books["large_thumbnail"]
)

CHROMA_DIR = "./chroma_db"

if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
    print("******************************************************************")
    print(f"Loading existing Chroma database from '{CHROMA_DIR}'...")
    print("******************************************************************")
    db_books = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    print("******************************************************************")
    print("Chroma database loaded.")
    print("******************************************************************")
else:   
    print("******************************************************************")
    print(f"Creating new Chroma database and embedding documents to '{CHROMA_DIR}'...")
    print("******************************************************************")
    raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db_books = Chroma.from_documents(
        documents,
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        persist_directory=CHROMA_DIR
    )
    print("******************************************************************")
    print("Chroma database created and documents embedded.")
    print("******************************************************************")
    # db_books.persist()



def retrieve_semantic_recommendations(
    query:str,
    category:str = None,
    tone:str = None,
    initial_top_k:int = 50,
    final_top_k:int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    book_list = [recs.page_content.strip().split()[0] for recs in recs]
    books_recs = books[books["isbn13"].astype(str).isin(book_list)].head(final_top_k)
    
    if category!= "All":
        books_recs = books_recs[books_recs["simple_categories"] == category][:final_top_k]
    else:
        books_recs = books_recs.head(final_top_k)
        
    if tone == "Happy":
        books_recs.sort_values(by = "joy", ascending=False, inplace=True)
    elif tone == "Sad":
        books_recs.sort_values(by = "sadness", ascending=False, inplace=True)
    elif tone == "Angry":
        books_recs.sort_values(by = "anger", ascending=False, inplace=True)
    elif tone == "Surprising":
        books_recs.sort_values(by = "surprise", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        books_recs.sort_values(by = "fear", ascending=False, inplace=True)
        
    return books_recs


def recommend_books(
    query: str,
    category: str = "All",
    tone: str = "Happy",
) -> pd.DataFrame:
    recommendations = retrieve_semantic_recommendations(
        query=query,
        category=category,
        tone=tone
    )
    
    results = []
    
    for _, row in recommendations.iterrows():
        description = row["description"]
        turncated_desc_split = description.split()
        turncated_description = " ".join(turncated_desc_split[:20]) + "..."
        
        authors_split = row["authors"].split(";")
        if len(authors_split) > 1:
            authors_str = f"{', '.join(authors_split[:-1])} and {authors_split[-1]}"
        else:
            authors_str = authors_split[0]
            
        caption = f"{row['title']} by {authors_str}: {turncated_description}"
        results.append((row["large_thumbnail"], caption))
        
    return results


categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tone = ["All"] + ["Happy", "Sad", "Angry", "Surprising", "Suspenseful"]
with gr.Blocks(theme=gr.themes.Soft(), title="Book Recommendation Dashboard") as dashboard: # Changed theme for a softer look
    gr.Markdown(
        """
        # 📚 Find Your Next Literary Adventure!
        Discover amazing books tailored to your **mood** and **interests**.
        """
    )

    with gr.Group(): # Grouping input elements for visual coherence
        with gr.Column():
            query = gr.Textbox(
                label="What are you looking for?",
                placeholder="e.g., 'a thrilling mystery with a strong female lead', 'lighthearted fantasy about dragons', 'historical fiction set in ancient Rome'",
                lines=2, # Allow for multi-line input
                interactive=True,
                scale=3 # Give more horizontal space to the query
            )

            with gr.Row(scale=1): # Row for dropdowns
                category = gr.Dropdown(
                    label="Genre", # Renamed label for better clarity
                    choices=categories,
                    value="All",
                    interactive=True
                )
                tone = gr.Dropdown(
                    label="Desired Tone", # Renamed label
                    choices=tone,
                    value="All",
                    interactive=True
                )
        
        submit_button = gr.Button(
            "✨ Get Personalized Recommendations ✨", 
            variant="primary", # Make the button more prominent
            size="lg" # Larger button size
        )

    gr.Markdown("---") # Horizontal line for separation
    gr.Markdown("## Your Book Recommendations")

    recommendations = gr.Gallery(
        label="Recommended Books", # Added a hint
        show_label=True,
        elem_id="gallery",
        columns=5, # Reduced columns for larger, more visible covers
        rows=2,
        object_fit="contain", # 'contain' or 'scale-down' often works better for book covers
        height="auto", # Let Gradio handle height based on content
        allow_preview=True, # Allow clicking on images for a larger view
        
    )
    
    # Clearer loading message
    loading_text = gr.Markdown("Fetching recommendations, please wait...", visible=False)

    submit_button.click(
        fn=lambda: gr.update(visible=True), # Show loading text
        inputs=None,
        outputs=loading_text,
        queue=False # Don't queue this immediate update
    ).then(
        fn=recommend_books,
        inputs=[query, category, tone],
        outputs=recommendations
    ).then(
        fn=lambda: gr.update(visible=False), # Hide loading text
        inputs=None,
        outputs=loading_text,
        queue=False
    )
    
if __name__ == "__main__":
    dashboard.launch()
    