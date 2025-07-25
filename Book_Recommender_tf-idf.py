import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
# Make sure 'cleaned_books.csv' is in the same directory as your Python script,
# or provide the full path to the file.
df = pd.read_csv('cleaned_books.csv')

# Fill any missing descriptions with an empty string to avoid errors during TF-IDF vectorization
df['description'] = df['description'].fillna('')

# Initialize the TF-IDF Vectorizer
# TfidfVectorizer converts a collection of raw documents to a matrix of TF-IDF features.
# 'stop_words="english"' removes common English words (like "the", "a", "is")
# that don't carry much meaning for similarity.
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the description column to create the TF-IDF matrix
# `fit_transform` learns the vocabulary and IDF (Inverse Document Frequency) from the text data
# and then transforms the text into numerical TF-IDF vectors.
tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'])

# Calculate the cosine similarity matrix
# `cosine_similarity` computes the cosine similarity between all pairs of books based on their TF-IDF vectors.
# The result is a square matrix where each cell [i, j] represents the similarity between book i and book j.
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a Series that maps book titles to their index in the DataFrame
# This allows us to quickly get the numerical index of a book given its title.
# `drop_duplicates()` is used in case there are multiple books with the exact same title,
# it ensures we only take the first occurrence.
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Define a function to get book recommendations
def get_recommendations(title, cosine_sim_matrix=cosine_sim, df=df, indices=indices):
    """
    Generates book recommendations based on cosine similarity of descriptions.

    Args:
        title (str): The title of the book for which to find recommendations.
        cosine_sim_matrix (np.array): The pre-calculated cosine similarity matrix.
        df (pd.DataFrame): The DataFrame containing book data.
        indices (pd.Series): A Series mapping book titles to their DataFrame indices.

    Returns:
        pd.Series: A Series of recommended book titles, or an error message if the book is not found.
    """
    # Check if the requested book title exists in our index
    if title not in indices:
        return f"Book '{title}' not found in the dataset. Please check the title."

    # Get the index of the book that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all books with that book
    # This retrieves the row corresponding to the chosen book's similarity scores with all other books.
    # `enumerate` adds an index to each similarity score, so we know which book it refers to.
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))

    # Sort the books based on the similarity scores in descending order
    # The `key=lambda x: x[1]` ensures we sort by the similarity score (the second element of each tuple).
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar books (excluding the book itself)
    # The most similar book will always be the book itself (score of 1), so we skip it by starting at index 1.
    # We take up to 11 elements to get 10 recommendations after skipping the first.
    sim_scores = sim_scores[1:11]

    # Get the actual DataFrame indices of the recommended books
    book_indices = [i[0] for i in sim_scores]

    # Return the titles of the top 10 most similar books
    return df['title'].iloc[book_indices]

# Example usage:
# You can change 'Gilead' to any other book title present in your 'cleaned_books.csv' file.
example_book_title = 'Harry potter'
recommendations = get_recommendations(example_book_title)

print(f"Recommendations for '{example_book_title}':")
print(recommendations)