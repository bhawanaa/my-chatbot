import os
import time
import openai
import requests
import pandas as pd
import docx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

from websockets.client import connect  # ✅ async version

import json
import wave
import ssl
import certifi

# Load .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not OPENAI_API_KEY or not ELEVENLABS_API_KEY:
    raise ValueError("Missing API keys in .env")

openai.api_key = OPENAI_API_KEY
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")

# App setup
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.urandom(24))
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global
UPLOAD_FOLDER = "uploads"
DIAGRAM_FOLDER = "static/diagrams"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DIAGRAM_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'xlsx', 'txt', 'jpg', 'jpeg', 'png', 'tif'}

global_embeddings = OpenAIEmbeddings()
global_vector_store = None

# === Utils ===

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(file_path, ext):
    try:
        if ext == "pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            text = "\n".join([doc.page_content for doc in docs])
            if not text:
                images = convert_from_path(file_path)
                text = "\n".join(pytesseract.image_to_string(img) for img in images)
        elif ext == "docx":
            doc = docx.Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif ext == "txt":
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
        elif ext in {"jpg", "jpeg", "png", "tif"}:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
        elif ext == "xlsx":
            sheets = pd.read_excel(file_path, sheet_name=None)
            text = "\n\n".join(
                f"{name}:\n{df.to_string(index=False)}" for name, df in sheets.items()
            )
        else:
            return "Unsupported file"
        return text
    except Exception as e:
        return f"Error: {e}"

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    return splitter.split_text(text)

def text_to_speech(text, output_path="static/output.mp3", voice_id="9BWtsMINqrJLrRacOk9x"):
    import requests

    # Define the endpoint and parameters
    url = "https://api.elevenlabs.io/v1/text-to-speech/JBFqnCBsd6RMkjVDRZzb/stream?output_format=mp3_44100_128"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2"
    }

    # Make the POST request with streaming enabled
    response = requests.post(url, headers=headers, json=data, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Audio saved to {output_path}")
    else:
        print("Error:", response.status_code, response.text)
    return output_path


import asyncio
from websockets import connect
import ssl
import certifi
import wave
import os
import json

async def text_to_speech_streaming(text, output_path="static/output.wav", voice_id="9BWtsMINqrJLrRacOk9x"):
    url = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

    if os.path.exists(output_path):
        os.remove(output_path)

    ssl_context = ssl.create_default_context(cafile=certifi.where())

    request_data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",  # ✅ compatible model for Aria
        "voice_settings": {
            "stability": 0.4,
            "similarity_boost": 0.75
        }
    }

    headers = [("xi-api-key", ELEVENLABS_API_KEY)]

    try:
        async with connect(url, additional_headers=headers, ssl=ssl_context, max_size=None) as ws:
            await ws.send(json.dumps(request_data))

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with wave.open(output_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(22050)

                while True:
                    try:
                        chunk = await ws.recv()
                        if isinstance(chunk, bytes):
                            wf.writeframes(chunk)
                        elif isinstance(chunk, str):
                            response = json.loads(chunk)
                            if response.get("audio_stream_chunk", {}).get("is_final"):
                                break
                    except Exception as e:
                        print("WebSocket closed:", e)
                        break

        print(f"✅ TTS saved to: {output_path}")
        return output_path

    except Exception as e:
        print("❌ Streaming TTS error:", e)
        return None
    
import aiohttp
import asyncio
import os
import json


async def text_to_speech_streaming_new(
    text, 
    output_path="static/output.mp3", 
    voice_id="9BWtsMINqrJLrRacOk9x"
):
    """
    Sends a POST request to ElevenLabs' streaming TTS endpoint and saves 
    the audio stream to output_path (in MPEG/MP3 format by default).
    """
    # The correct endpoint for streaming (note: it's HTTPS, not WSS)
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

    # Remove any existing file to avoid confusion
    if os.path.exists(output_path):
        os.remove(output_path)

    # JSON body sent to ElevenLabs
    request_data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",  # or whichever model is appropriate
        "voice_settings": {
            "stability": 0.4,
            "similarity_boost": 0.75
        }
    }

    # Required headers
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        #"Accept": "audio/mpeg",
        "Content-Type": "application/json"
        
    }

    # Create the output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=request_data) as response:
            if response.status != 200:
                print(f"❌ Streaming TTS error: HTTP {response.status}")
                error_msg = await response.text()
                print("Response text:", error_msg)
                return None

            # Write the streamed audio chunks to a file
            with open(output_path, "wb") as f:
                async for chunk in response.content.iter_chunked(4096):
                    if chunk:
                        f.write(chunk)

    print(f"✅ TTS saved to: {output_path}")
    return output_path

# Example usage:
# asyncio.run(text_to_speech_streaming("Hello from ElevenLabs!"))



def extract_images_from_pdf(pdf_path):
    pages = convert_from_path(pdf_path)
    paths = []
    for i, page in enumerate(pages):
        out = os.path.join(DIAGRAM_FOLDER, f"page_{i+1}.png")
        page.save(out, "PNG")
        paths.append(out)
    return paths

# === Routes ===

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(400, detail="Unsupported file")

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    ext = filename.rsplit(".", 1)[1].lower()
    text = extract_text(file_path, ext)
    if text.startswith("Error") or not text.strip():
        raise HTTPException(400, detail=text)

    chunks = split_text(text)

    global global_vector_store
    if global_vector_store is None:
        global_vector_store = FAISS.from_texts(chunks, global_embeddings)
    else:
        global_vector_store.add_texts(chunks)

    response = {"message": f"{filename} processed"}
    if ext == "pdf":
        images = extract_images_from_pdf(file_path)
        response["diagram_urls"] = [f"/static/diagrams/{os.path.basename(p)}" for p in images]

    return JSONResponse(content=response)

import asyncio
from fastapi.responses import JSONResponse

async def handle_request(answer: str):
    # Start the TTS conversion as a background task.
    # You can also use a task queue like Celery for heavier processing.
    asyncio.create_task(text_to_speech_streaming_new(answer))
    
    # Return response immediately.
    return JSONResponse(content={
        "answer": answer,
        "audio_url": f"/static/output.mp3?nocache={int(time.time())}"
    })


@app.post("/ask")
async def ask_question(request: Request, question: str = Form(...)):
    if global_vector_store is None:
        return JSONResponse(content={"answer": "Upload documents first"}, status_code=400)

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer based on the context below.

Context:
{context}

Question: {input}
Answer:
""")

    retriever = global_vector_store.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    doc_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)

    try:
        result = retrieval_chain.invoke({"input": question})
        answer = result.get("answer", "No answer found.")
    except Exception as e:
        print("❌ Error:", e)
        answer = "An error occurred."


    print(f"🤖 Answer: {answer}")
    text_to_speech(answer)
    #await handle_request(answer=answer)
    # await text_to_speech_streaming_new(answer)
    return JSONResponse(content={
        "answer": answer,
        "audio_url": f"/static/output.mp3?nocache={int(time.time())}"
    })

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    ext = file.filename.split('.')[-1].lower()
    if ext not in {'mp3', 'wav', 'm4a'}:
        raise HTTPException(400, detail="Audio format not supported")

    temp_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    with open(temp_path, "rb") as audio_file:
        result = openai.Audio.transcribe("whisper-1", audio_file)

    return JSONResponse(content={"transcript": result.get("text", "")})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)
