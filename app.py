import streamlit as st
import os
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Load spaCy model for text processing
nlp = spacy.load('en_core_web_sm')

# Streamlit application
st.title("Smart PDF Q&A Assistant 🤖")
st.markdown("**Upload multiple PDF files and let the Smart PDF Assistant analyze their content to answer all your questions intelligently and efficiently!**")


# File uploader for multiple PDFs
pdf_files = st.file_uploader("Upload PDF files", type='pdf', accept_multiple_files=True)

if pdf_files:
    raw_text = ''

    # Loop through the uploaded PDF files and extract text from each
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                raw_text += content

    st.subheader("Extracted Text Preview:")
    st.write(raw_text[:1000])  # Display first 1000 characters of extracted text for preview

    # Split the extracted text into sentences
    doc = nlp(raw_text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 50]  # Filter out short sentences



    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)

    # User input for the question
    query = st.text_input("Ask a question about the PDF content:")

    if query:
        # Convert the query into a vector
        query_vector = vectorizer.transform([query])

        # Compute cosine similarity between the query and sentences
        similarities = cosine_similarity(query_vector, sentence_vectors).flatten()

        # Get the maximum similarity score
        max_similarity = similarities.max()

        # Set a threshold for considering the question relevant
        similarity_threshold = 0.1  # Adjust this threshold based on your needs

        if max_similarity < similarity_threshold:
            st.subheader("Response:")
            st.write("I'm sorry, but I don't have relevant information regarding that question.")
        else:
            # Get the indices of the top N similar sentences
            top_n_indices = similarities.argsort()[-3:][::-1]  # Get top 3 most similar sentences
            best_answers = [sentences[i] for i in top_n_indices]

            st.subheader("Answer:")
            for answer in best_answers:
                st.write(answer)  # Display the top 3 answers
