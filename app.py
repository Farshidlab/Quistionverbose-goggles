from flask import Flask, render_template, request
import openai
import os
import faiss
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = 'Api Key'

# Load the precomputed document vectors into the vector database (Faiss index)
dimension = 300  # Dimensionality of the document vectors
index_path = 'vector_db/index.faiss'  # Path to the Faiss index

# Load the Faiss index if it exists
if os.path.exists(index_path):
    index = faiss.read_index(index_path)
else:
    index = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/answer', methods=['GET','POST'])
def answer_question():
    # Get the question from the request
    question = request.form['question']

    # Get the document path from the request
    document_path = request.form['document_path']

    # Extract text content from the PDF document
    document_text = extract_text_from_pdf(document_path)

    # Retrieve relevant documents from the vector database (Faiss index)
    if index is not None:
        document_vectors = compute_vectors(document_text)  # Replace with your own vector computation logic
        k = 5  # Number of documents to retrieve
        _, document_indices = index.search(document_vectors, k)
        selected_documents = [get_document_path(index) for index in document_indices]
    else:
        # If the vector database is not available, fallback to retrieving all documents in the uploads folder
        selected_documents = get_all_documents()

    # Perform question-answering using OpenAI's API on the selected documents
    answers = []
    for doc_path in selected_documents:
        with open(doc_path, 'r') as file:
            document_text = file.read()
        response = openai.Completion.create(
            model="davinci",
            prompt=f"Question: {question}\nAnswer:",
            text=document_text,
            max_tokens=100
        )
        answer = response.choices[0].text.strip()
        answers.append(answer)

    # Render the result template with the answers
    return render_template('result.html', answers=answers)

if __name__ == '__main__':
    app.run()

def compute_vectors(document_path):
    # Implement your own logic to compute vectors for the given document text using the chosen vectorization method
    # Return the document vectors
    with open(document_path, 'r') as file:
        document_text = file.read()

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Compute the TF-IDF vectors for the document
    document_vector = vectorizer.fit_transform([document_text])

    # Return the document vector as a numpy array
    return document_vector.toarray()[0]

def get_document_path(index):
    # Retrieve the document path for the given index from your data structure
    # Return the document path
    document_paths = [
        'C:/Users/mdfar/Downloads/'        
        # Add more document paths as needed
    ]
    return document_paths[index]

def get_all_documents():
    # Get all document paths from the uploads folder
    upload_folder = 'uploads/'
    all_documents = [os.path.join(upload_folder, filename) for filename in os.listdir(upload_folder)]
    return all_documents

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page in range(pdf_reader.numPages):
            text += pdf_reader.getPage(page).extractText()
        return text
