import os
import json
import time
from dotenv import load_dotenv
import pandas as pd
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
import docx  # for DOCX files
from fastapi.staticfiles import StaticFiles
import httpx


http_client = httpx.Client(verify=False)

# Initialize FastAPI app
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up session middleware (similar to Flask sessions)
SECRET_KEY = os.urandom(24)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# Set up Jinja2 templates (make sure you have a "templates" directory with index.html)
templates = Jinja2Templates(directory="templates")

# Pandas configuration
pd.set_option('display.max_rows', None)

# Configure uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'xlsx', 'docx', 'txt'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("No OPENAI_API_KEY found in the environment.")

# Set the API key (if required by your libraries)
os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize global embeddings and vector store
global_embeddings = OpenAIEmbeddings()
global_vector_store = None  # Will be initialized on first file upload

def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route – renders index.html with chat history
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    chat_history = request.session.get("chat_history", [])
    return templates.TemplateResponse("index.html", {"request": request, "chat_history": chat_history})

# Upload endpoint – accepts file upload and processes it
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file or not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="No valid file provided")
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    # Save the uploaded file to disk
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)
    
    file_ext = filename.rsplit(".", 1)[1].lower()
    chunks = process_file_into_chunks(file_path, file_ext)
    if isinstance(chunks, str):  # if an error message was returned
        raise HTTPException(status_code=400, detail=chunks)
    
    global global_vector_store
    if global_vector_store is None:
        global_vector_store = FAISS.from_texts(chunks, global_embeddings)
    else:
        global_vector_store.add_texts(chunks)
    
    return JSONResponse(content={"message": f"File '{filename}' processed and added to the knowledge base."})

def lookup_on_web(question: str) -> str:
    """
    Fallback function: if no answer is found in the documents,
    simulate a web search using ChatOpenAI.
    """
    web_search_prompt = f"Search the web for: {question}\nProvide a concise answer."
    web_chain = ChatOpenAI(api_key=openai_api_key, http_client=http_client)
    result = web_chain.invoke({"input": web_search_prompt})
    return result.get("answer", "Sorry, no results found on the web.")

# Ask endpoint – accepts a question and returns an answer
@app.post("/ask")
async def ask_question(request: Request, question: str = Form(...)):
    if not question:
        raise HTTPException(status_code=400, detail="No question provided")
    if global_vector_store is None:
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")
    
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context.
    <context>
    {context}
    </context>
    Question: {input}""")
    
    retriever = global_vector_store.as_retriever()
    document_chain = create_stuff_documents_chain(ChatOpenAI(api_key=openai_api_key, http_client=http_client), prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    result = retrieval_chain.invoke({"input": question})
    if not result or "I do not have enough context" in result:
        fallback_answer = lookup_on_web(question)
        answer_text = fallback_answer + "\n\n[This answer was fetched from the web as a fallback.]"
    else:
        answer_text = result.get("answer", "")
    
    # Update session chat history
    chat_history = request.session.get("chat_history", [])
    chat_history.append({"type": "question", "text": question})
    chat_history.append({"type": "answer", "text": answer_text})
    request.session["chat_history"] = chat_history
    
    return JSONResponse(content={"answer": answer_text, "chat_history": chat_history})

def process_file_into_chunks(file_path: str, file_ext: str):
    """
    Process a file based on its type and return a list of text chunks.
    """
    try:
        if file_ext == "pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            text = "\n".join([doc.page_content for doc in docs])
        elif file_ext == "xlsx":
            sheets = pd.read_excel(file_path, engine="openpyxl", sheet_name=None)
            sheet_texts = []
            for sheet_name, df in sheets.items():
                print(df)
                # Identify columns starting with "DragForce"
                drag_cols = [col for col in df.columns if str(col).startswith("DragForce")]
                print("Printing DragForce columns:")
                print(drag_cols)
                if drag_cols:
                    df = df.dropna(subset=drag_cols, how='all')
                # Identify columns starting with "YieldForce" and drop rows with all missing values
                other_cols = [col for col in df.columns if str(col).startswith("YieldForce")]
                if other_cols:
                    df = df.dropna(subset=other_cols, how='all')
                sheet_text = f"Sheet: {sheet_name}\n" + df.to_string(index=False, max_rows=len(df))
                sheet_texts.append(sheet_text)
            text = "\n\n".join(sheet_texts)
            print("Final text:")
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)
