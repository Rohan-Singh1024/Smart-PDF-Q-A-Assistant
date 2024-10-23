# Smart PDF Q&A Assistant 🤖

This project is a **Streamlit** application that allows users to upload multiple PDF files and ask questions about the content. The app uses **spaCy** for text processing and **TF-IDF** with **cosine similarity** to find relevant answers to the user's questions based on the uploaded PDF content.

## Features
- Upload multiple PDF files.
- Extract and display the content of the PDFs.
- Ask questions about the PDF content using natural language input.
- The system processes the PDFs, splits the text into sentences, and uses a TF-IDF Vectorizer to compute cosine similarity between the question and the document sentences.
- The app retrieves the most relevant answers based on the similarity score.

## How It Works
1. Users can upload one or more PDF files.
2. The text from each PDF is extracted and processed using **spaCy** for sentence splitting.
3. The app uses **TF-IDF** to convert the sentences into vectors.
4. Users can ask questions, which are also converted into vectors.
5. **Cosine similarity** is computed between the question vector and the sentence vectors.
6. The app returns the top 3 most relevant sentences as answers to the user's question.

## Technology Stack
- **Streamlit**: Front-end interface and web app framework.
- **PyPDF2**: For extracting text from PDF files.
- **spaCy**: For natural language processing and splitting text into sentences.
- **Scikit-learn**: To apply the **TF-IDF** vectorizer and calculate **cosine similarity**.

## Setup

### 1. Install the requirements
Clone the repository and install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
