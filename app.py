import os
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

# Additional imports for OCR and diagram extraction
import pytesseract
from pdf2image import convert_from_path
from PIL import Image  # For image file processing

# Import openai for Whisper speech recognition
import openai

from gtts import gTTS

# This module is imported so that we can 
# play the converted audio
import os

load_dotenv()

# Set Tesseract command from environment variable or default to Linux path
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
print("Tesseract command set to:", pytesseract.pytesseract.tesseract_cmd)

# Initialize HTTP client for ChatOpenAI
http_client = httpx.Client(verify=False)

# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up session middleware (similar to Flask sessions)
SECRET_KEY = os.urandom(24)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# Set up Jinja2 templates (ensure you have a "templates" directory with index.html)
templates = Jinja2Templates(directory="templates")

# Pandas configuration
pd.set_option('display.max_rows', None)

# Configure uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'xlsx', 'docx', 'txt', 'png', 'jpg', 'jpeg', 'bmp', 'gif', 'tif', 'tiff'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("No OPENAI_API_KEY found in the environment.")

# Set the API key for OpenAI
openai.api_key = openai_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize global embeddings and vector store
global_embeddings = OpenAIEmbeddings()
global_vector_store = None  # Will be initialized on first file upload

app.mount("/static", StaticFiles(directory="static"), name="static")

def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route – renders index.html with chat history
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    chat_history = request.session.get("chat_history", [])
    return templates.TemplateResponse("index.html", {"request": request, "chat_history": chat_history})

# Upload endpoint – accepts file upload, processes text, and extracts diagrams for PDFs
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
    
    # Process text extraction for all file types
    chunks = process_file_into_chunks(file_path, file_ext)
    if isinstance(chunks, str):  # if an error message was returned
        raise HTTPException(status_code=400, detail=chunks)
    
    global global_vector_store
    if global_vector_store is None:
        global_vector_store = FAISS.from_texts(chunks, global_embeddings)
    else:
        global_vector_store.add_texts(chunks)
    
    response_data = {"message": f"File '{filename}' processed and added to the knowledge base."}
    
    # If the file is a PDF, also extract diagrams/images
    if file_ext == "pdf":
        image_paths = extract_diagrams_from_pdf(file_path)
        image_urls = [f"/static/diagrams/{os.path.basename(path)}?nocache={int(time.time())}" for path in image_paths]
        print(image_urls)
        response_data["diagram_urls"] = image_urls
    
    return JSONResponse(content=response_data)

def lookup_on_web(question: str) -> str:
    """
    Fallback function: if no answer is found in the documents,
    simulate a web search using ChatOpenAI.
    """
    web_search_prompt = f"Search the web for: {question}\nProvide a concise answer."
    web_chain = ChatOpenAI(api_key=openai_api_key, http_client=http_client)
    result = web_chain.invoke({"input": web_search_prompt})
    return result.get("answer", "Sorry, no results found on the web.")

@app.post("/ask")
async def ask_question(request: Request, question: str = Form(...)):
    if not question:
        raise HTTPException(status_code=400, detail="No question provided")

    if global_vector_store is None:
        return JSONResponse(content={"answer": "No documents uploaded yet."}, status_code=400)

    # Define the prompt
    prompt = ChatPromptTemplate.from_template("""
    <context>
    {context}
    </context>
    Question: {input}
    """)

    retriever = global_vector_store.as_retriever()
    document_chain = create_stuff_documents_chain(ChatOpenAI(api_key=openai_api_key, http_client=http_client), prompt)

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    try:
        result = retrieval_chain.invoke({"input": question})
        answer_text = result.get("answer", "Sorry, I couldn't find an answer.")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        answer_text = "An error occurred while fetching the answer."

    print(f"🔍 User Question: {question}")
    print(f"🤖 AI Answer: {answer_text}")

    # Generate speech from AI response using pyttsx3
    #audio_file = text_to_speech(answer_text, output_path="static/output.wav")
    audio_file = text_to_speech_g(answer_text, output_path="static/output.wav")

    if not os.path.exists(audio_file):
        return JSONResponse(content={"answer": answer_text, "error": "Audio file not found"}, status_code=500)

    return JSONResponse(content={
        "answer": answer_text,
        "audio_url": f"/static/output.wav?nocache={int(time.time())}"
    })

def process_file_into_chunks(file_path: str, file_ext: str):
    """
    Process a file based on its type and return a list of text chunks.
    This function now includes OCR handling for PDFs that have no extractable text,
    and support for image file types.
    """
    try:
        if file_ext == "pdf":
            # Attempt to extract text using PyPDFLoader
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            text = "\n".join([doc.page_content for doc in docs]).strip()
            # If no text is extracted, assume it's an image-based PDF and use OCR
            if not text:
                print("No text extracted using PyPDFLoader, attempting OCR...")
                images = convert_from_path(file_path)
                ocr_text_list = []
                for image in images:
                    ocr_text = pytesseract.image_to_string(image)
                    ocr_text_list.append(ocr_text)
                text = "\n".join(ocr_text_list)
                if not text.strip():
                    return "OCR failed to extract any text from the PDF."
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
                    for col in drag_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce').abs()
                    df = df.dropna(subset=drag_cols, how='all')
                
                # Identify columns starting with "YieldForce" and drop rows with all missing values.
                other_cols = [col for col in df.columns if str(col).startswith("YF")]
                if other_cols:
                    for col in other_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce').abs()
                    df = df.dropna(subset=other_cols, how='all')
                
                sheet_text = f"Sheet: {sheet_name}\n" + df.to_string(index=False, max_rows=len(df))
                sheet_texts.append(sheet_text)
            text = "\n\n".join(sheet_texts)
        elif file_ext == "docx":
            document = docx.Document(file_path)
            text = "\n".join([para.text for para in document.paragraphs])
        elif file_ext == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif file_ext in {"png", "jpg", "jpeg", "bmp", "gif", "tif", "tiff"}:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
        else:
            return f"Unsupported file type: {file_ext}"
    except Exception as e:
        return f"Error processing file: {str(e)}"
    
    if not text.strip():
        return "No text could be extracted from the file."
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    chunks = [chunk for chunk in chunks if chunk.strip()]
    if not chunks:
        return "The document contains no extractable text after splitting."
    return chunks

# import pyttsx3

# def text_to_speech(response_text, output_path="output.wav"):
#     output_dir = "static"
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     output_path = os.path.join(output_dir, "output.wav")
#     print("Access the file at:", output_path)
#     engine = pyttsx3.init()
    
#     # List all available voices for inspection
#     voices = engine.getProperty('voices')
#     print("Available voices:")
#     # for voice in voices:
#     #     print(f"ID: {voice.id} | Name: {voice.name}")
#     #     if "English (America, New York City)" in voice.name:
#     #         engine.setProperty('voice', voice.id)
#     #         print(f"Selected voice: {voice.name}")
#     #         break

#     # # Attempt to select a male voice based on common male voice names
#     # male_voice_id = None
#     # for voice in voices:
#     #     # You can adjust these keywords depending on what your system offers
#     #     if any(keyword in voice.name.lower() for keyword in ["david", "alex", "male", "george"]):
#     #         male_voice_id = voice.id
#     #         break

#     # if male_voice_id:
#     #     engine.setProperty('voice', male_voice_id)
#     #     print(f"Using male voice: {male_voice_id}")
#     # else:
#     #     print("No male voice found; using default voice.")

#     # Adjust the speech rate if desired
#     engine.setProperty('rate', 180)
#     print(response_text)

#     engine.save_to_file(response_text, output_path)
#     #engine.say(response_text)
#     engine.runAndWait()
#     if os.path.exists(output_path):
#         print("File created at:", output_path)
#         output_url = f"{output_path}?nocache={int(time.time())}"
#         print("Access the file at:", output_url)
#         return output_url
#     else:
#         print("Audio file not found at:", output_path)
#         return None
#     #print("Access the file at:", output_url)
#     return output_url
#     #return output_path



def text_to_speech_g(response_text, output_path="output.wav"):

    # Language in which you want to convert
    language = 'en'

    # Passing the text and language to the engine, 
    # here we have marked slow=False. Which tells 
    # the module that the converted audio should 
    # have a high speed
    myobj = gTTS(text=response_text, lang=language, slow=False)

    output_dir = "static"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "output.wav")

    #print(output_path)

    # Saving the converted audio in a mp3 file named
    # welcome 
    myobj.save(output_path)

    # Playing the converted file
    os.system("play welcome.mp3 tempo 3.0")
    return output_path


def extract_diagrams_from_pdf(pdf_path, output_folder="static/diagrams"):
    """
    Extracts images from a PDF by converting each page to an image.
    Returns a list of file paths for the extracted images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    pages = convert_from_path(pdf_path)
    image_paths = []
    for i, page in enumerate(pages):
        image_path = os.path.join(output_folder, f"page_{i+1}.png")
        page.save(image_path, "PNG")
        image_paths.append(image_path)
    return image_paths

# # Endpoint to extract diagrams only
# @app.post("/upload_diagrams")
# async def upload_diagrams(file: UploadFile = File(...)):
#     if not file or not file.filename.lower().endswith(".pdf"):
#         raise HTTPException(status_code=400, detail="Please upload a valid PDF file.")
    
#     filename = secure_filename(file.filename)
#     file_path = os.path.join(UPLOAD_FOLDER, filename)
    
#     contents = await file.read()
#     with open(file_path, "wb") as f:
#         f.write(contents)
    
#     # Extract images (diagrams) from the PDF
#     image_paths = extract_diagrams_from_pdf(file_path)
#     image_urls = [f"/static/diagrams/{os.path.basename(path)}?nocache={int(time.time())}" for path in image_paths]
    
#     return JSONResponse(content={
#         "message": f"Diagrams extracted from '{filename}'.",
#         "diagram_urls": image_urls
#     })

# New endpoint for speech recognition using the Whisper API
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'm4a', 'mp4', 'ogg'}
    if not file or file.filename.split('.')[-1].lower() not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid audio file. Allowed types: wav, mp3, m4a, mp4, ogg")
    
    tmp_filename = secure_filename(file.filename)
    tmp_path = os.path.join(UPLOAD_FOLDER, tmp_filename)
    contents = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(contents)
    
    try:
        with open(tmp_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {e}")
    
    return JSONResponse(content={"transcript": transcript.get("text", "")})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)
