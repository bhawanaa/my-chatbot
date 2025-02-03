import os
import json
import time
from dotenv import load_dotenv
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

pd.set_option('display.max_rows', None)


# Configure Uploads: supports pdf, xlsx, docx, and txt
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'xlsx', 'docx', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("No OPENAI_API_KEY found in the environment.")

# Optionally, set the API key in os.environ (if your libraries expect it there)
os.environ["OPENAI_API_KEY"] = openai_api_key
# Set OpenAI API Key
#os.environ["OPENAI_API_KEY"] = "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"

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

def lookup_on_web(question):
    """
    Fallback function to search the web if no answer is found in the documents.
    Here we use ChatOpenAI to simulate a web search. In a real-world scenario,
    integrate with a dedicated web search API (e.g., SerpAPI).
    """
    web_search_prompt = f"Search the web for: {question}\nProvide a concise answer."
    web_chain = ChatOpenAI()  # You can pass additional parameters if needed.
    result = web_chain.invoke({"input": web_search_prompt})
    return result.get("answer", "Sorry, no results found on the web.")

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
    Answer the following question based only on the provided context.
    <context>
    {context}
    </context>

    Question: {input}""")
    
    # Build Retrieval Chain using the global vector store
    retriever = global_vector_store.as_retriever()
    document_chain = create_stuff_documents_chain(ChatOpenAI(), prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    response = retrieval_chain.invoke({"input": question})

    # Fallback: if the answer is empty or seems not useful, then search the web.
    if not response or "I do not have enough context" in response:
        fallback_answer = lookup_on_web(question)
        response = fallback_answer + "\n\n[This answer was fetched from the web as a fallback.]"

    
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
        # elif file_ext == "xlsx":
        #     sheets = pd.read_excel(file_path, engine="openpyxl", sheet_name=None)
        #     #text = "\n".join(df.to_string(index=False) for df in sheets.values())
         
        #     text = "\n\n".join(df.to_string(index=False, max_rows=len(df)) for df in sheets.values())
        #     print(text)
        elif file_ext == "xlsx":
            sheets = pd.read_excel(file_path, engine="openpyxl", sheet_name=None)
            sheet_texts = []
            for sheet_name, df in sheets.items():
                #df = df.fillna(0)

                print(df)
                
                # Identify columns starting with "DragForce"
                drag_cols = [col for col in df.columns if str(col).startswith("DragForce")]
                print('printting...')
                print(drag_cols)
                if drag_cols:
                    df = df.dropna(subset=drag_cols, how='all')
                # Identify other columns
                other_cols = [col for col in df.columns  if str(col).startswith("YieldForce")]
                if other_cols:
                    df = df.dropna(subset=other_cols, how='all')
                # rest_cols = [col for col in df.columns  if col not in drag_cols and col not in other_cols and not str(col).startswith("Legal")]
                # print(rest_cols)
                
                # drag_text = ""
                # other_text = ""
                # if drag_cols:
                #     drag_text = "Drag Force Data:\n" + df[drag_cols].to_string(index=False, max_rows=len(df), na_rep="0")
                # if other_cols:
                #     other_text = "Other Data:\n" + df[other_cols].to_string(index=False, max_rows=len(df), na_rep="0")
                # if rest_cols:
                #     rest_text = "Rest Data:\n" + df[rest_cols].to_string(index=False, max_rows=len(df))
                # sheet_text = f"Sheet: {sheet_name}\n" + drag_text + "\n" + other_text +"\n" + rest_text
                # print(sheet_text)
                # sheet_texts.append(sheet_text)
                sheet_text = f"Sheet: {sheet_name}\n" + df.to_string(index=False, max_rows=len(df))
                sheet_texts.append(sheet_text)

            text = "\n\n".join(sheet_texts)
            print('final..')
            print(text)
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks



if __name__ == '__main__':
    app.run(debug=True)
