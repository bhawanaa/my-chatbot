import os
import json
import time
import pandas as pd
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
import docx  # for DOCX files

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Needed for session storage

# Configure Uploads: supports pdf, xlsx, docx, and txt
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'xlsx', 'docx', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Initialize a global vector store and embeddings
global_embeddings = OpenAIEmbeddings()
global_vector_store = None  # Will initialize on first upload

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    """Render the homepage with chat history."""
    if "chat_history" not in session:
        session["chat_history"] = []
    return render_template("index.html", chat_history=session["chat_history"])

@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Endpoint to upload a file and add its content to the global vector store.
    This can be called multiple times to build your knowledge base.
    """
    file = request.files.get("file")
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "No valid file provided"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    file_ext = filename.rsplit(".", 1)[1].lower()
    chunks = process_file_into_chunks(file_path, file_ext)
    if isinstance(chunks, str):  # if an error message was returned
        return jsonify({"error": chunks}), 400

    # Update the global vector store
    global global_vector_store
    if global_vector_store is None:
        global_vector_store = FAISS.from_texts(chunks, global_embeddings)
    else:
        global_vector_store.add_texts(chunks)

    return jsonify({"message": f"File '{filename}' processed and added to the knowledge base."}), 200

@app.route("/ask", methods=["POST"])
def ask_question():
    """
    Ask a question against the aggregated document store.
    """
    question = request.form.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    if global_vector_store is None:
        return jsonify({"error": "No documents uploaded yet."}), 400

    # Create Prompt Template
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")
    
    # Build Retrieval Chain using the global vector store
    retriever = global_vector_store.as_retriever()
    document_chain = create_stuff_documents_chain(ChatOpenAI(), prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    response = retrieval_chain.invoke({"input": question})
    
    # Update chat history in session
    if "chat_history" not in session:
        session["chat_history"] = []
    session["chat_history"].append({"type": "question", "text": question})
    session["chat_history"].append({"type": "answer", "text": response["answer"]})
    session.modified = True

    return jsonify({"answer": response["answer"], "chat_history": session["chat_history"]})

def process_file_into_chunks(file_path, file_ext):
    """
    Process a file based on its type and return a list of text chunks.
    """
    try:
        if file_ext == "pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            # For PDFs, assume each page (or section) is a doc
            text = "\n".join([doc.page_content for doc in docs])
        elif file_ext == "xlsx":
            sheets = pd.read_excel(file_path, engine="openpyxl", sheet_name=None)
            text = "\n".join(df.to_string(index=False) for df in sheets.values())
        elif file_ext == "docx":
            document = docx.Document(file_path)
            text = "\n".join([para.text for para in document.paragraphs])
        elif file_ext == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            return f"Unsupported file type: {file_ext}"
    except Exception as e:
        return f"Error processing file: {str(e)}"

    # Split the full text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

if __name__ == '__main__':
    app.run(debug=True)
