from langchain_openai import ChatOpenAI
import os
import time
import openai
import requests
import pandas as pd
import docx
import httpx  # For HTTP requests to Ollama
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.cors import CORSMiddleware
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
import aiohttp
import asyncio
import edge_tts
import azure.cognitiveservices.speech as speechsdk
import io
import base64
from openai import AsyncOpenAI
import rapidfuzz
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not OPENAI_API_KEY or not ELEVENLABS_API_KEY:
    raise ValueError("Missing API keys in .env")

openai.api_key = OPENAI_API_KEY

# Configure Tesseract for Windows - try common installation paths
try:
    # Try different possible Tesseract paths on Windows
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe", 
        r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.getenv("USERNAME", "")),
        os.getenv("TESSERACT_CMD", "tesseract")  # Try from PATH or env var
    ]
    
    tesseract_found = False
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            tesseract_found = True
            print(f"✅ Tesseract found at: {path}")
            break
    
    if not tesseract_found:
        print("⚠️  Tesseract not found. OCR functionality will be limited.")
        # Set a default that might work if tesseract is in PATH
        pytesseract.pytesseract.tesseract_cmd = "tesseract"
        
except Exception as e:
    print(f"⚠️  Tesseract configuration error: {e}. OCR functionality will be limited.")

# App setup
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.urandom(24))
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
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

# Global document storage and search
document_texts = []  # Store all document chunks
document_keywords = []  # Store key terms extracted from documents
semantic_model = None  # Will be initialized when needed

def initialize_semantic_model():
    """Initialize the sentence transformer model for semantic search."""
    global semantic_model
    if semantic_model is None:
        try:
            semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Semantic model initialized")
        except Exception as e:
            print(f"⚠️  Could not initialize semantic model: {e}")
    return semantic_model

def extract_key_terms(text, min_length=3):
    """Extract key terms from text for fuzzy matching."""
    # Remove common stop words and extract meaningful terms
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'within', 'without', 'against', 'upon', 'beneath', 'beside', 'beyond', 'across', 'toward', 'throughout', 'underneath', 'alongside'}
    
    # Extract words (remove punctuation, keep alphanumeric)
    words = re.findall(r'\b[a-zA-Z]{' + str(min_length) + ',}\b', text.lower())
    
    # Filter out stop words and return unique terms
    key_terms = list(set([word for word in words if word not in stop_words]))
    return key_terms

def fuzzy_search_documents(query, threshold=60):
    """Search documents using fuzzy matching to handle speech recognition errors."""
    global document_keywords, document_texts
    
    if not document_texts:
        return []
    
    query_words = extract_key_terms(query)
    matched_chunks = []
    
    for i, doc_keywords in enumerate(document_keywords):
        max_score = 0
        best_matches = []
        
        for query_word in query_words:
            # Find best fuzzy match for this query word in document keywords
            matches = process.extractBests(query_word, doc_keywords, scorer=fuzz.ratio, limit=3)
            for match, score in matches:
                if score >= threshold:
                    max_score = max(max_score, score)
                    best_matches.append((match, score))
        
        if max_score >= threshold:
            matched_chunks.append({
                'text': document_texts[i],
                'score': max_score,
                'matches': best_matches,
                'index': i
            })
    
    # Sort by relevance score
    matched_chunks.sort(key=lambda x: x['score'], reverse=True)
    return matched_chunks[:5]  # Return top 5 matches

def semantic_search_documents(query, top_k=3):
    """Search documents using semantic similarity."""
    global document_texts, semantic_model
    
    if not document_texts or not semantic_model:
        return []
    
    try:
        # Encode query and documents
        query_embedding = semantic_model.encode([query])
        doc_embeddings = semantic_model.encode(document_texts)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Minimum similarity threshold
                results.append({
                    'text': document_texts[idx],
                    'score': float(similarities[idx]),
                    'index': int(idx)
                })
        
        return results
    except Exception as e:
        print(f"⚠️  Semantic search error: {e}")
        return []

def intelligent_document_search(query):
    """Combine fuzzy and semantic search for optimal document retrieval."""
    # Try fuzzy search first (handles speech recognition errors)
    fuzzy_results = fuzzy_search_documents(query, threshold=70)
    
    # Try semantic search for better context understanding
    semantic_results = semantic_search_documents(query, top_k=3)
    
    # Combine and deduplicate results
    all_results = {}
    
    # Add fuzzy results
    for result in fuzzy_results:
        idx = result['index']
        all_results[idx] = {
            'text': result['text'],
            'fuzzy_score': result['score'],
            'semantic_score': 0,
            'matches': result.get('matches', [])
        }
    
    # Add semantic results
    for result in semantic_results:
        idx = result['index']
        if idx in all_results:
            all_results[idx]['semantic_score'] = result['score']
        else:
            all_results[idx] = {
                'text': result['text'],
                'fuzzy_score': 0,
                'semantic_score': result['score'],
                'matches': []
            }
    
    # Calculate combined score and sort
    final_results = []
    for idx, result in all_results.items():
        combined_score = (result['fuzzy_score'] * 0.4 + result['semantic_score'] * 100 * 0.6)
        final_results.append({
            'text': result['text'],
            'score': combined_score,
            'fuzzy_score': result['fuzzy_score'],
            'semantic_score': result['semantic_score'],
            'matches': result['matches']
        })
    
    # Sort by combined score and return top 3
    final_results.sort(key=lambda x: x['score'], reverse=True)
    return final_results[:3]

# WebSocket connection manager for real-time chat
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Chat history storage (in production, use a database)
chat_history = []

# Conversation state management
conversation_sessions = {}

class ConversationSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.context = []
        self.is_active = False
        self.last_activity = time.time()
        self.voice_settings = {
            "voice_id": "9BWtsMINqrJLrRacOk9x",  # Default to Aria
            "speed": 1.0
        }
    
    def add_message(self, role: str, content: str):
        self.context.append({"role": role, "content": content, "timestamp": time.time()})
        self.last_activity = time.time()
        
        # Keep only last 10 messages for context
        if len(self.context) > 10:
            self.context = self.context[-10:]
    
    def get_context_string(self):
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.context[-5:]])
    
    def is_expired(self):
        return time.time() - self.last_activity > 300  # 5 minutes timeout

# === Utils ===

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(file_path, ext):
    try:
        if ext == "pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            text = "\n".join([doc.page_content for doc in docs])
            if not text.strip():
                # Try OCR as fallback for image PDFs
                try:
                    images = convert_from_path(file_path)
                    ocr_text = ""
                    for img in images:
                        try:
                            ocr_text += pytesseract.image_to_string(img) + "\n"
                        except Exception as ocr_e:
                            print(f"⚠️  OCR failed for image: {ocr_e}")
                            ocr_text += "[Image content could not be extracted - Tesseract OCR not available]\n"
                    text = ocr_text if ocr_text.strip() else "PDF contains images but OCR is not available"
                except Exception as pdf_e:
                    print(f"⚠️  PDF image extraction failed: {pdf_e}")
                    text = "PDF processed but some image content may be missing (OCR unavailable)"
        elif ext == "docx":
            doc = docx.Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif ext == "txt":
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
        elif ext in {"jpg", "jpeg", "png", "tif"}:
            try:
                img = Image.open(file_path)
                text = pytesseract.image_to_string(img)
                if not text.strip():
                    text = "Image processed but no text content detected"
            except Exception as img_e:
                print(f"⚠️  Image OCR failed: {img_e}")
                text = "Image uploaded but OCR is not available - please install Tesseract OCR for text extraction from images"
        elif ext == "xlsx":
            sheets = pd.read_excel(file_path, sheet_name=None)
            text = "\n\n".join(
                f"{name}:\n{df.to_string(index=False)}" for name, df in sheets.items()
            )
        else:
            return "Unsupported file"
        return text
    except Exception as e:
        return f"Error processing file: {e}"

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    return splitter.split_text(text)

def text_to_speech(text, output_path="static/output.mp3", voice_id="zcAOhNBS3c14rBihAFp1"):
    import requests

    # Define the endpoint and parameters
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream?output_format=mp3_44100_128"
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

async def text_to_speech_with_options(
    text, 
    voice_id="9BWtsMINqrJLrRacOk9x",
    speed=1.0,
    output_path="static/output.mp3"
):
    """
    Enhanced TTS with voice and speed options
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

    if os.path.exists(output_path):
        os.remove(output_path)

    request_data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.4,
            "similarity_boost": 0.75,
            "speed": speed
        }
    }

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=request_data) as response:
            if response.status != 200:
                print(f"❌ Enhanced TTS error: HTTP {response.status}")
                return None

            with open(output_path, "wb") as f:
                async for chunk in response.content.iter_chunked(4096):
                    if chunk:
                        f.write(chunk)

    print(f"✅ Enhanced TTS saved to: {output_path}")
    return output_path

# 🆓 FREE ALTERNATIVE: OpenAI TTS (much cheaper than ElevenLabs!)
async def openai_text_to_speech(
    text,
    voice="alloy",  # alloy, echo, fable, onyx, nova, shimmer
    speed=1.0,
    output_path="static/output.mp3"
):
    """
    OpenAI TTS - High quality, much cheaper alternative to ElevenLabs
    $15 per 1M characters vs ElevenLabs $330 per 1M characters!
    """
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        if os.path.exists(output_path):
            os.remove(output_path)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        response = await client.audio.speech.create(
            model="tts-1-hd",  # High-definition model
            voice=voice,
            input=text,
            speed=speed
        )

        # Save the audio file
        response.stream_to_file(output_path)
        
        print(f"✅ OpenAI TTS saved to: {output_path} (Voice: {voice}, Speed: {speed}x)")
        return output_path

    except Exception as e:
        print(f"❌ OpenAI TTS error: {e}")
        return None

# 🆓 FREE ALTERNATIVE: Edge TTS (Completely Free!)
async def edge_text_to_speech(
    text,
    voice="en-US-AriaNeural",  # Many free voices available
    speed=1.0,
    output_path="static/output.mp3"
):
    """
    Microsoft Edge TTS - Completely free, unlimited usage!
    Requires: pip install edge-tts
    """
    try:
        import edge_tts
        
        if os.path.exists(output_path):
            os.remove(output_path)

        # Create directory if it doesn't exist (but only if there's a directory part)
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create if there's actually a directory path
            os.makedirs(output_dir, exist_ok=True)

        # Adjust speed for Edge TTS format
        rate = f"{int((speed - 1) * 50):+d}%"

        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(output_path)
        
        print(f"✅ Edge TTS saved to: {output_path} (Voice: {voice}, Speed: {speed}x)")
        return output_path

    except ImportError:
        print("❌ Edge TTS not installed. Run: pip install edge-tts")
        return None
    except Exception as e:
        print(f"❌ Edge TTS error: {e}")
        return None

# 🆓 FREE ALTERNATIVE: Azure Speech Services (Free tier: 500k chars/month)
async def azure_text_to_speech(
    text,
    voice="en-US-AriaNeural",
    speed=1.0,
    output_path="static/output.wav"
):
    """
    Azure Speech Services - 500,000 characters free per month
    Requires: pip install azure-cognitiveservices-speech
    Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in .env
    """
    try:
        import azure.cognitiveservices.speech as speechsdk
        
        azure_key = os.getenv("AZURE_SPEECH_KEY")
        azure_region = os.getenv("AZURE_SPEECH_REGION")
        
        if not azure_key or not azure_region:
            print("❌ Azure Speech key/region not found in .env")
            return None

        if os.path.exists(output_path):
            os.remove(output_path)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        speech_config = speechsdk.SpeechConfig(subscription=azure_key, region=azure_region)
        speech_config.speech_synthesis_voice_name = voice
        
        # Create SSML with speed control
        ssml = f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
            <voice name="{voice}">
                <prosody rate="{speed}">
                    {text}
                </prosody>
            </voice>
        </speak>
        """

        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=speechsdk.audio.AudioOutputConfig(filename=output_path)
        )

        result = synthesizer.speak_ssml_async(ssml).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"✅ Azure TTS saved to: {output_path} (Voice: {voice}, Speed: {speed}x)")
            return output_path
        else:
            print(f"❌ Azure TTS failed: {result.reason}")
            return None

    except ImportError:
        print("❌ Azure Speech SDK not installed. Run: pip install azure-cognitiveservices-speech")
        return None
    except Exception as e:
        print(f"❌ Azure TTS error: {e}")
        return None

# 🗣️ TEXT PREPROCESSING - Fix pronunciation issues
def fix_pronunciation(text, voice_id=""):
    """
    Fix common pronunciation issues in TTS
    """
    # Determine voice type
    is_italian = "it-IT" in voice_id
    is_spanish = "es-" in voice_id or "Diego" in voice_id
    
    # For Spanish voices, use Spanish phonetics  
    if is_spanish:
        replacements = {
            "Veloci AI": "Veloce Ah Ee",
            "Veloci": "Veloce", 
            "AI": "Ah Ee",
            " ai ": " Ah Ee ",
            " AI.": " Ah Ee.",
            " AI,": " Ah Ee,",
            " AI ": " Ah Ee ",
            "A.I.": "Ah Ee",
            "A.I": "Ah Ee",
        }
    else:
        # Common AI/ML term corrections for other voices
        replacements = {
            # Try different phonetic approaches
            "Veloci AI": "Velocity Aye Aye" if not is_italian else "Veh-LOH-chee Ah Ee",
            "Veloci": "Velocity" if not is_italian else "Veh-LOH-chee", 
            "AI": "Aye Aye",  # Simple "Aye Aye" approach
            " ai ": " Aye Aye ",
            " AI.": " Aye Aye.",
            " AI,": " Aye Aye,",
            " AI ": " Aye Aye ",
            "A.I.": "Aye Aye",
            "A.I": "Aye Aye",
        }
    
    # Common tech terms (same for all languages)
    common_terms = {
        "ML": "Em Ell",
        "API": "Ay Pee Eye", 
        "UI": "You Eye",
        "URL": "You Arr Ell",
        "PDF": "Pee Dee Eff",
        "CSV": "See Ess Vee",
        "JSON": "Jay-son",
        "XML": "Ex Em Ell",
    }
    
    replacements.update(common_terms)
    
    # Italian-specific replacements
    if is_italian:
        replacements.update({
            "Veloci AI": "Veloce A I",  # Override with Italian specific
            "Veloci": "Veloce",  # More natural Italian
            "AI": "A I",  # Simple A I for Italian
            " ai ": " A I ",
            " AI.": " A I.",
            " AI,": " A I,", 
            " AI ": " A I ",
            "A.I.": "A I",
            "A.I": "A I",
            "Hello": "Ciao",     # Use Italian greetings when appropriate
        })
    
    # Apply replacements with multiple strategies
    import re
    processed_text = text
    
    # First pass - Handle specific combinations first (most specific to least specific)
    specific_patterns = [
        (r'\bveloci\s+ai\b', replacements.get("Veloci AI", "Veloci AI")),
        (r'\bveloci\s*ai\b', replacements.get("Veloci AI", "Veloci AI")),
    ]
    
    for pattern, replacement in specific_patterns:
        processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
    
    # Second pass - individual terms (but skip if already processed)
    individual_replacements = {k: v for k, v in replacements.items() if "AI" not in k}
    for original, replacement in individual_replacements.items():
        processed_text = re.sub(r'\b' + re.escape(original) + r'\b', replacement, processed_text, flags=re.IGNORECASE)
    
    # Third pass - word boundary replacements for stubborn cases
    if is_spanish:
        word_patterns = [
            (r'\bveloci\s*ai\b', "Veloce Ah Ee"),
            (r'\bveloci\b', "Veloce"),
            (r'\bai\b', "Ah Ee"),
            (r'(?<!\w)ai(?!\w)', "Ah Ee"),
            (r'\bA\.?I\.?\b', "Ah Ee"),
        ]
    elif is_italian:
        word_patterns = [
            (r'\bveloci\s*ai\b', "Veloce A I"),  # Very simple for Italian
            (r'\bveloci\b', "Veloce"),
            (r'\bai\b', "A I"),  # Just A space I
            (r'(?<!\w)ai(?!\w)', "A I"),
            (r'\bA\.?I\.?\b', "A I"),
        ]
    else:
        word_patterns = [
            (r'\bveloci\s*ai\b', "Velocity Aye Aye"),
            (r'\bveloci\b', "Velocity"),
            (r'\bai\b', "Aye Aye"),
            (r'(?<!\w)ai(?!\w)', "Aye Aye"),
            (r'\bA\.?I\.?\b', "Aye Aye"),
        ]
    
    for pattern, replacement in word_patterns:
        processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
    
    return processed_text

# 🤖 FREE AI PROVIDERS - Support for open-source models
async def get_ai_response_free(message, provider="ollama", model="llama3.2:3b"):
    """
    Get AI response from free/open-source providers with intelligent document search
    
    Providers:
    - ollama: Local AI models (free, private)
    - groq: Fast cloud AI (free tier)  
    - huggingface: Open models (free tier)
    
    Features:
    - Normal conversation capabilities
    - Enhanced document search when documents are available
    - Fuzzy matching for speech recognition errors (e.g., "drga" → "drag")
    - Semantic search for context understanding
    """
    
    # Check if we have documents and if this looks like a document-related query
    global document_texts
    enhanced_prompt = message
    
    # Only use document search if the query is substantive and document-related (contains 'document', 'file', or is long)
    if document_texts and ("document" in message.lower() or "file" in message.lower() or len(message.split()) > 5):
        search_results = intelligent_document_search(message)
        if search_results and search_results[0]['score'] > 0.3:
            context_parts = []
            for i, result in enumerate(search_results[:2]):
                context_parts.append(f"Relevant document content:\n{result['text'][:400]}")
            document_context = "\n\n".join(context_parts)
            enhanced_prompt = f"You are a helpful AI assistant. The user has uploaded documents, and here's some potentially relevant content:\n\n{document_context}\n\nUser question: {message}\n\nPlease provide a helpful response. If the question relates to the document content above, use it to enhance your answer. For general questions, greetings, or other topics, respond normally."
    
    if provider == "ollama":
        try:
            # Check if Ollama is running
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model,
                        "prompt": enhanced_prompt,
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "Sorry, I couldn't generate a response.")
                else:
                    return f"Ollama error: {response.status_code}"
                    
        except Exception as e:
            return f"Ollama not available. Install with: 'curl -fsSL https://ollama.ai/install.sh | sh' and run 'ollama run {model}'. Error: {str(e)}"
    
    elif provider == "groq":
        # Groq API (fast, free tier available)
        try:
            async with httpx.AsyncClient() as client:
                groq_api_key = os.getenv("GROQ_API_KEY")
                if not groq_api_key:
                    return "Please set GROQ_API_KEY environment variable. Get free key from: https://console.groq.com/"
                
                request_data = {
                    "model": model,  # e.g., "llama3-8b-8192"
                    "messages": [{"role": "user", "content": enhanced_prompt}],
                    "temperature": 0.7
                }
                
                print(f"🔍 GROQ DEBUG: API Key present: {bool(groq_api_key)}")
                print(f"🔍 GROQ DEBUG: Model: {model}")
                print(f"🔍 GROQ DEBUG: Request data: {request_data}")
                
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {groq_api_key}",
                        "Content-Type": "application/json"
                    },
                    json=request_data
                )
                
                print(f"🔍 GROQ DEBUG: Response status: {response.status_code}")
                print(f"🔍 GROQ DEBUG: Response content: {response.text[:500]}...")
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    return f"Groq API error: {response.status_code} - {response.text[:200]}..."
                    
        except Exception as e:
            print(f"🔍 GROQ DEBUG: Exception: {str(e)}")
            return f"Groq API error: {str(e)}"
    
    elif provider == "huggingface":
        # Hugging Face Inference API
        try:
            async with httpx.AsyncClient() as client:
                hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
                if not hf_api_key:
                    return "Please set HUGGINGFACE_API_KEY environment variable. Get free key from: https://huggingface.co/settings/tokens"
                
                response = await client.post(
                    f"https://api-inference.huggingface.co/models/{model}",
                    headers={
                        "Authorization": f"Bearer {hf_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "inputs": enhanced_prompt,
                        "parameters": {
                            "temperature": 0.7,
                            "max_new_tokens": 500
                        }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get("generated_text", "No response generated")
                    return str(result)
                else:
                    return f"Hugging Face API error: {response.status_code}"
                    
        except Exception as e:
            return f"Hugging Face API error: {str(e)}"
    
    return f"Unknown provider: {provider}"

# � OPENAI VOICE MODELS - Advanced voice capabilities
async def openai_speech_to_text(audio_data):
    """
    Convert speech to text using OpenAI Whisper API
    More accurate than Web Speech API, especially for technical terms
    """
    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Create audio file from data
        audio_file = io.BytesIO(audio_data)
        audio_file.name = "audio.wav"
        
        # Use Whisper for transcription
        transcript = await client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json"  # Get confidence scores
        )
        
        return {
            "text": transcript.text,
            "confidence": getattr(transcript, 'confidence', 0.9),  # Whisper is typically high confidence
            "language": getattr(transcript, 'language', 'en')
        }
        
    except Exception as e:
        return {"error": f"OpenAI Whisper error: {str(e)}"}

async def openai_text_to_speech(text, voice="alloy", speed=1.0, output_path="static/output.mp3"):
    """
    Convert text to speech using OpenAI TTS API
    Voices: alloy, echo, fable, onyx, nova, shimmer
    """
    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        response = await client.audio.speech.create(
            model="tts-1-hd",  # High quality model
            voice=voice,
            input=text,
            speed=speed
        )
        
        # Get the audio content directly
        audio_data = response.content
            
        with open(output_path, "wb") as f:
            f.write(audio_data)
            
        return output_path
        
    except Exception as e:
        print(f"❌ OpenAI TTS error: {e}")
        return None

# �🎛️ UNIFIED TTS FUNCTION - Switch between different providers
async def universal_text_to_speech(
    text,
    voice_id="elevenlabs:9BWtsMINqrJLrRacOk9x",  # Format: provider:voice_id
    speed=1.0,
    output_path="static/output.mp3"
):
    """
    Universal TTS function that can use different providers
    
    Providers:
    - elevenlabs: High quality, expensive ($330/1M chars)
    - openai: High quality, cheap ($15/1M chars) 
    - edge: Free, unlimited, good quality
    - azure: Free tier (500k chars/month), excellent quality
    
    Voice ID format: "provider:voice_name"
    Examples:
    - "edge:en-US-AriaNeural"
    - "openai:alloy" 
    - "azure:en-US-AriaNeural"
    - "elevenlabs:9BWtsMINqrJLrRacOk9x"
    """
    
    # Parse provider and voice from voice_id
    if ":" in voice_id:
        provider, voice = voice_id.split(":", 1)
    else:
        # Fallback for legacy voice IDs (assume ElevenLabs)
        provider = "elevenlabs"
        voice = voice_id
    
    print(f"🎤 TTS Request - Provider: {provider}, Voice: {voice}, Text: '{text[:50]}...'")  # Debug log
    
    # Fix pronunciation issues
    processed_text = fix_pronunciation(text, voice_id)
    if processed_text != text:
        print(f"🗣️  Fixed pronunciation: '{text}' → '{processed_text}'")
    
    # Calculate cost per 1000 characters
    cost_per_1000 = {
        "elevenlabs": 0.33,   # $330 per 1M = $0.33 per 1k
        "openai": 0.015,      # $15 per 1M = $0.015 per 1k
        "edge": 0.0,          # Free
        "azure": 0.0          # Free (within limits)
    }.get(provider, 0.0)
    
    cost = (len(text) / 1000) * cost_per_1000
    
    # Route to appropriate TTS function
    try:
        if provider == "elevenlabs":
            audio_path = await text_to_speech_with_options(processed_text, voice, speed, output_path)
        elif provider == "openai":
            audio_path = await openai_text_to_speech(processed_text, voice, speed, output_path)
        elif provider == "edge":
            audio_path = await edge_text_to_speech(processed_text, voice, speed, output_path)
        elif provider == "azure":
            audio_path = await azure_text_to_speech(processed_text, voice, speed, output_path)
        else:
            print(f"❌ Unknown TTS provider: {provider}")
            return None, 0.0
            
        return audio_path, cost
    except Exception as e:
        print(f"❌ TTS error with {provider}: {e}")
        return None, 0.0

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
    return templates.TemplateResponse("index_interactive.html", {"request": request})

@app.get("/classic", response_class=HTMLResponse)
async def classic_interface(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/conversation", response_class=HTMLResponse)
async def conversation_mode(request: Request):
    return templates.TemplateResponse("conversation.html", {"request": request})

# WebSocket endpoint for real-time chat
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data["type"] == "question":
                question = message_data["content"]
                
                # Process the question and get response
                if global_vector_store is None:
                    response = {"type": "error", "content": "Upload documents first"}
                else:
                    try:
                        # Send typing indicator
                        await manager.send_personal_message(
                            json.dumps({"type": "typing", "content": "AI is typing..."}), 
                            websocket
                        )
                        
                        # Process the question
                        prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer based on the context below.
Answer concisely and conversationally.

Context:
{context}

Question: {input}
Answer:
""")
                        
                        retriever = global_vector_store.as_retriever()
                        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
                        
                        doc_chain = create_stuff_documents_chain(llm, prompt)
                        retrieval_chain = create_retrieval_chain(retriever, doc_chain)
                        
                        result = retrieval_chain.invoke({"input": question})
                        answer = result.get("answer", "No answer found.")
                        
                        # Store in chat history
                        chat_entry = {
                            "timestamp": time.time(),
                            "question": question,
                            "answer": answer
                        }
                        chat_history.append(chat_entry)
                        
                        response = {
                            "type": "answer",
                            "content": answer,
                            "timestamp": chat_entry["timestamp"]
                        }
                        
                    except Exception as e:
                        response = {"type": "error", "content": f"An error occurred: {str(e)}"}
                
                await manager.send_personal_message(json.dumps(response), websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Conversational WebSocket endpoint for natural voice conversations
@app.websocket("/conversation")
async def conversation_websocket(websocket: WebSocket):
    await websocket.accept()
    
    try:
        # Initialize conversation session
        session_id = f"conv_{int(time.time())}_{id(websocket)}"
        conversation_sessions[session_id] = ConversationSession(session_id)
        session = conversation_sessions[session_id]
        
        await websocket.send_text(json.dumps({
            "type": "session_start",
            "session_id": session_id,
            "message": "🎤 Conversation mode activated! Start speaking naturally."
        }))
        
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            message_type = message_data.get("type")
            
            print(f"🔍 DEBUG: Received WebSocket message: {message_type} - {data[:100]}...")
            
            if message_type == "voice_input":
                # User spoke something
                user_message = message_data.get("content", "")
                voice_settings = message_data.get("voice_settings", session.voice_settings)
                
                # Update voice settings
                session.voice_settings.update(voice_settings)
                
                # Add user message to context
                session.add_message("user", user_message)
                
                # Send processing indicator
                await websocket.send_text(json.dumps({
                    "type": "processing",
                    "message": "🤔 Thinking..."
                }))
                
                try:
                    # Choose AI provider (check for free provider preference)
                    ai_provider = message_data.get("ai_provider", "openai")  # Default to OpenAI
                    ai_model = message_data.get("ai_model", "gpt-3.5-turbo")
                    
                    # Generate response with conversation context
                    if ai_provider in ["ollama", "groq", "huggingface"]:
                        # Use free AI providers
                        context_prompt = f"""
You are an intelligent AI assistant with expertise in engineering, physics, and technical subjects. Answer questions naturally and confidently based on your knowledge.

Previous conversation:
{session.get_context_string()}

Question: {user_message}

Provide a direct, helpful answer. Don't mention documents or sources - just answer as if you know the information.
"""
                        ai_response = await get_ai_response_free(context_prompt, ai_provider, ai_model)
                        
                    elif global_vector_store is not None:
                        # Context-aware response with document knowledge (OpenAI)
                        context_prompt = f"""
You are an expert AI assistant with comprehensive knowledge. Answer questions confidently and naturally.

Previous conversation:
{session.get_context_string()}

Question: {user_message}

Provide a clear, direct answer based on your knowledge. Be conversational and helpful.
"""
                        
                        retriever = global_vector_store.as_retriever()
                        llm = ChatOpenAI(model_name=ai_model, temperature=0.7)  # More conversational
                        
                        prompt = ChatPromptTemplate.from_template("""
You are an intelligent AI assistant with expertise in engineering, physics, and technical subjects. 

Use the following information to enhance your knowledge:
{context}

{conversation_prompt}

Provide a direct, confident answer. Don't mention sources or documents - respond as if this is your inherent knowledge:
""")
                        
                        doc_chain = create_stuff_documents_chain(llm, prompt)
                        retrieval_chain = create_retrieval_chain(retriever, doc_chain)
                        
                        result = retrieval_chain.invoke({
                            "input": user_message,
                            "conversation_prompt": context_prompt
                        })
                        ai_response = result.get("answer", "I'm not sure how to respond to that.")
                    
                    else:
                        # General conversation without documents (OpenAI)
                        llm = ChatOpenAI(model_name=ai_model, temperature=0.7)
                        
                        context_prompt = f"""
You are an intelligent AI assistant. Answer questions directly and confidently based on your knowledge.

Previous conversation:
{session.get_context_string()}

Question: {user_message}

Provide a helpful, natural response:
"""
                        
                        response = llm.invoke(context_prompt)
                        ai_response = response.content
                    
                    # Add AI response to context
                    session.add_message("assistant", ai_response)
                    
                    # Generate TTS for the response using universal TTS system
                    try:
                        # Use universal TTS with provider routing
                        audio_path, cost = await universal_text_to_speech(
                            ai_response, 
                            session.voice_settings["voice_id"], 
                            session.voice_settings.get("speed", 1.0)
                        )
                        audio_url = f"/static/output.mp3?nocache={int(time.time())}" if audio_path else None
                        print(f"✅ TTS generated successfully. Provider: {session.voice_settings['voice_id']}, Cost: ${cost:.6f}")
                    except Exception as e:
                        print(f"❌ TTS error: {e}")
                        audio_url = None
                    
                    # Send response back
                    await websocket.send_text(json.dumps({
                        "type": "ai_response",
                        "content": ai_response,
                        "audio_url": audio_url,
                        "voice_used": session.voice_settings["voice_id"],
                        "session_id": session_id
                    }))
                    
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "content": f"Sorry, I encountered an error: {str(e)}"
                    }))
            
            elif message_type == "voice_settings":
                # Update voice preferences
                session.voice_settings.update(message_data.get("settings", {}))
                await websocket.send_text(json.dumps({
                    "type": "settings_updated",
                    "voice_settings": session.voice_settings
                }))
            
            elif message_type == "end_conversation":
                # End conversation session
                if session_id in conversation_sessions:
                    del conversation_sessions[session_id]
                
                await websocket.send_text(json.dumps({
                    "type": "conversation_ended",
                    "message": "👋 Conversation ended. Thanks for chatting!"
                }))
                break
    
    except WebSocketDisconnect:
        # Clean up session
        if session_id in conversation_sessions:
            del conversation_sessions[session_id]
    except Exception as e:
        print(f"Conversation WebSocket error: {e}")

# Get chat history
@app.get("/chat/history")
async def get_chat_history():
    return JSONResponse(content={"history": chat_history[-10:]})  # Return last 10 messages

# Clear chat history
@app.post("/chat/clear")
async def clear_chat_history():
    global chat_history
    chat_history.clear()
    return JSONResponse(content={"message": "Chat history cleared"})

# Streaming response for long answers
@app.post("/ask/stream")
async def ask_question_stream(request: Request, question: str = Form(...)):
    if global_vector_store is None:
        return JSONResponse(content={"error": "Upload documents first"}, status_code=400)
    
    async def generate_response():
        try:
            prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer based on the context below.
Answer in a conversational and detailed manner.

Context:
{context}

Question: {input}
Answer:
""")
            
            retriever = global_vector_store.as_retriever()
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, streaming=True)
            
            doc_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, doc_chain)
            
            # Get the response in chunks for streaming
            result = retrieval_chain.invoke({"input": question})
            answer = result.get("answer", "No answer found.")
            
            # Simulate streaming by yielding chunks
            words = answer.split()
            for i, word in enumerate(words):
                yield f"data: {json.dumps({'chunk': word + ' ', 'done': False})}\n\n"
                await asyncio.sleep(0.05)  # Small delay for streaming effect
            
            yield f"data: {json.dumps({'chunk': '', 'done': True})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate_response(), media_type="text/plain")

# Suggest follow-up questions
@app.post("/suggest")
async def suggest_questions(last_answer: str = Form(...)):
    if global_vector_store is None:
        return JSONResponse(content={"suggestions": []})
    
    try:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        
        prompt = f"""
Based on this answer: "{last_answer}"
Generate 3 relevant follow-up questions that a user might ask.
Return only the questions, one per line, without numbering.
"""
        
        response = llm.invoke(prompt)
        suggestions = [q.strip() for q in response.content.split('\n') if q.strip()]
        
        return JSONResponse(content={"suggestions": suggestions[:3]})
        
    except Exception as e:
        return JSONResponse(content={"suggestions": []})

# Enhanced TTS with voice selection - Now supports FREE alternatives!
@app.post("/tts")
async def text_to_speech_api(
    text: str = Form(...), 
    voice_id: str = Form("openai:alloy"),  # Default to free OpenAI voice
    speed: float = Form(1.0)
):
    try:
        print(f"🎤 TTS API called with voice_id: {voice_id}, text: '{text[:50]}...'")  # Debug log
        # Use universal TTS function with provider:voice format
        output_path, cost = await universal_text_to_speech(text, voice_id, speed)
        
        # Parse provider for response
        provider = voice_id.split(":")[0] if ":" in voice_id else "elevenlabs"
        voice = voice_id.split(":", 1)[1] if ":" in voice_id else voice_id
        
        if output_path:
            return JSONResponse(content={
                "audio_url": f"/static/output.mp3?nocache={int(time.time())}",
                "provider_used": provider,
                "voice_used": voice,
                "cost_estimate": f"${cost:.6f}",
                "characters": len(text)
            })
        else:
            return JSONResponse(content={"error": "TTS generation failed"}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Smart document summary
@app.post("/summarize")
async def summarize_documents():
    if global_vector_store is None:
        return JSONResponse(content={"error": "No documents uploaded"}, status_code=400)
    
    try:
        # Get all documents
        docs = global_vector_store.similarity_search("", k=10)
        combined_text = " ".join([doc.page_content for doc in docs])
        
        # Limit text length for summarization
        if len(combined_text) > 4000:
            combined_text = combined_text[:4000] + "..."
        
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
        
        prompt = f"""
Provide a concise summary of the following documents in 3-4 sentences:

{combined_text}

Summary:
"""
        
        response = llm.invoke(prompt)
        summary = response.content
        
        return JSONResponse(content={"summary": summary})
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(400, detail="Unsupported file type. Supported: PDF, DOCX, XLSX, TXT, JPG, JPEG, PNG, TIF")

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    ext = filename.rsplit(".", 1)[1].lower()
    text = extract_text(file_path, ext)
    
    # Handle extraction errors more gracefully
    if text.startswith("Error processing file:"):
        raise HTTPException(400, detail=f"Could not process {filename}: {text}")
    elif not text.strip():
        # If no text was extracted, still allow upload but warn user
        text = f"File '{filename}' was uploaded but no text content was extracted. This may be an image file without OCR capability."

    chunks = split_text(text)
    
    # Calculate some stats
    word_count = len(text.split())
    chunk_count = len(chunks)

    # Update global search structures
    global global_vector_store, document_texts, document_keywords
    
    # Initialize semantic model if not already done
    initialize_semantic_model()
    
    # Add to document texts and extract keywords for fuzzy search
    document_texts.extend(chunks)
    for chunk in chunks:
        keywords = extract_key_terms(chunk)
        document_keywords.append(keywords)
    
    # Update FAISS vector store
    if global_vector_store is None:
        global_vector_store = FAISS.from_texts(chunks, global_embeddings)
    else:
        global_vector_store.add_texts(chunks)

    response = {
        "message": f"✅ {filename} processed successfully!",
        "stats": {
            "word_count": word_count,
            "chunk_count": chunk_count,
            "file_size": f"{os.path.getsize(file_path) / 1024:.1f} KB"
        }
    }
    
    # Special handling for PDFs with images
    if ext == "pdf":
        try:
            images = extract_images_from_pdf(file_path)
            response["diagram_urls"] = [f"/static/diagrams/{os.path.basename(p)}" for p in images]
            response["message"] += f" Found {len(images)} page(s) with potential diagrams."
        except Exception as e:
            print(f"Could not extract images from PDF: {e}")

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

def get_one_liner(answer: str) -> str:
    # Split by sentence end punctuation, then return the first non-empty sentence.
    import re
    sentences = re.split(r'[.!?]\s+', answer.strip())
    return sentences[0].strip() + '.' if sentences and sentences[0] else answer

def clean_answer(answer: str) -> str:
    # List common greetings to remove
    greetings = ["Sure thing!", "Absolutely!", "Of course!", "Okay!"]
    for greeting in greetings:
        if answer.strip().startswith(greeting):
            # Remove the greeting from the beginning
            answer = answer[len(greeting):].strip()
    # Optionally, remove any extra punctuation from the start and then get the first sentence of the remaining text
    sentences = answer.split('.')
    for sentence in sentences:
        trimmed = sentence.strip()
        if trimmed:
            return trimmed + '.'
    return answer


@app.post("/ask")
async def ask_question(
    request: Request, 
    question: str = Form(...),
    voice_id: str = Form("9BWtsMINqrJLrRacOk9x"),
    speed: float = Form(1.0)
):
    if global_vector_store is None:
        return JSONResponse(content={"answer": "Upload documents first"}, status_code=400)

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer based on the context below.
Answer concisely and conversationally. Be friendly and engaging.

Context:
{context}

Question: {input}
Answer:
""")

    retriever = global_vector_store.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

    doc_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)

    try:
        result = retrieval_chain.invoke({"input": question})
        answer = result.get("answer", "No answer found.")
        
        # Store in chat history
        chat_entry = {
            "timestamp": time.time(),
            "question": question,
            "answer": answer
        }
        chat_history.append(chat_entry)
        
        # Keep only last 50 messages in history
        if len(chat_history) > 50:
            chat_history.pop(0)
            
    except Exception as e:
        print("❌ Error:", e)
        answer = "An error occurred while processing your question."

    print(f"🤖 Answer: {answer}")
    one_liner_answer = clean_answer(answer)
    print(one_liner_answer)
    
    # Generate TTS with selected voice and speed using FREE alternatives!
    try:
        # Use universal TTS function for cost-effective generation
        audio_path, cost = await universal_text_to_speech(one_liner_answer, voice_id, speed)
        provider = voice_id.split(":")[0] if ":" in voice_id else "elevenlabs"
        print(f"✅ TTS generated with {provider}, Cost: ${cost:.6f}, Speed: {speed}x")
    except Exception as e:
        print(f"TTS Error: {e}")
        # Fallback to basic TTS
        try:
            text_to_speech(one_liner_answer, voice_id=voice_id)
        except Exception as fallback_error:
            print(f"Fallback TTS Error: {fallback_error}")
    
    return JSONResponse(content={
        "answer": one_liner_answer,
        "full_answer": answer,
        "audio_url": f"/static/output.mp3?nocache={int(time.time())}",
        "timestamp": chat_entry["timestamp"],
        "voice_used": voice_id,
        "speed_used": speed
    })

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    ext = file.filename.split('.')[-1].lower()
    if ext not in {'mp3', 'wav', 'm4a', 'webm', 'ogg'}:
        raise HTTPException(400, detail="Audio format not supported. Supported: MP3, WAV, M4A, WebM, OGG")

    temp_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        with open(temp_path, "rb") as audio_file:
            result = openai.Audio.transcribe("whisper-1", audio_file)
        
        transcript = result.get("text", "")
        
        # Clean up temp file
        os.remove(temp_path)
        
        return JSONResponse(content={
            "transcript": transcript,
            "message": "Audio transcribed successfully!"
        })
        
    except Exception as e:
        # Clean up temp file even if error occurs
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(500, detail=f"Transcription failed: {str(e)}")

# OpenAI Whisper Speech-to-Text endpoint
@app.post("/openai-whisper")
async def openai_whisper_transcription(audio_file: UploadFile = File(...)):
    """
    Use OpenAI Whisper for speech-to-text transcription
    Much more accurate than Web Speech API for technical terms
    """
    try:
        # Read audio file data
        audio_data = await audio_file.read()
        
        # Use OpenAI Whisper
        result = await openai_speech_to_text(audio_data)
        
        if "error" in result:
            raise HTTPException(500, detail=result["error"])
            
        return JSONResponse(content={
            "success": True,
            "transcript": result["text"],
            "confidence": result.get("confidence", 0.9),
            "language": result.get("language", "en"),
            "provider": "openai-whisper"
        })
        
    except Exception as e:
        raise HTTPException(500, detail=f"OpenAI Whisper transcription failed: {str(e)}")

# Get available voice options - Including FREE alternatives!
@app.get("/voices")
async def get_available_voices():
    voices = [
        # 🚀 OPENAI TTS - Much cheaper than ElevenLabs, excellent quality!
        {"id": "openai:alloy", "name": "Alloy (OpenAI)", "description": "🚀 Natural, friendly voice", "provider": "openai", "cost_per_1000_chars": 0.015},
        {"id": "openai:echo", "name": "Echo (OpenAI)", "description": "🚀 Expressive, dynamic voice", "provider": "openai", "cost_per_1000_chars": 0.015},
        {"id": "openai:fable", "name": "Fable (OpenAI)", "description": "🚀 Warm, engaging voice", "provider": "openai", "cost_per_1000_chars": 0.015},
        {"id": "openai:nova", "name": "Nova (OpenAI)", "description": "🚀 Professional female voice", "provider": "openai", "cost_per_1000_chars": 0.015},
        {"id": "openai:onyx", "name": "Onyx (OpenAI)", "description": "🚀 Deep, authoritative male", "provider": "openai", "cost_per_1000_chars": 0.015},
        {"id": "openai:shimmer", "name": "Shimmer (OpenAI)", "description": "🚀 Bright, energetic voice", "provider": "openai", "cost_per_1000_chars": 0.015},
        
        # 🆓 COMPLETELY FREE - Edge TTS (Unlimited!)
        {"id": "edge:en-US-AriaNeural", "name": "Aria Edge (FREE)", "description": "🆓 Microsoft neural voice - Unlimited", "provider": "edge", "cost_per_1000_chars": 0.0},
        {"id": "edge:en-US-JennyNeural", "name": "Jenny Edge (FREE)", "description": "🆓 Friendly female - Unlimited", "provider": "edge", "cost_per_1000_chars": 0.0},
        {"id": "edge:en-US-GuyNeural", "name": "Guy Edge (FREE)", "description": "🆓 Professional male - Unlimited", "provider": "edge", "cost_per_1000_chars": 0.0},
        {"id": "edge:en-US-DavisNeural", "name": "Davis Edge (FREE)", "description": "🆓 Authoritative male - Unlimited", "provider": "edge", "cost_per_1000_chars": 0.0},
        
        # 🇮🇹 ITALIAN VOICES - Edge TTS (FREE)
        {"id": "edge:it-IT-ElsaNeural", "name": "Elsa Italian (FREE)", "description": "🇮🇹 Natural Italian female - Unlimited", "provider": "edge", "cost_per_1000_chars": 0.0},
        {"id": "edge:it-IT-IsabellaNeural", "name": "Isabella Italian (FREE)", "description": "🇮🇹 Professional Italian female - Unlimited", "provider": "edge", "cost_per_1000_chars": 0.0},
        {"id": "edge:it-IT-DiegoNeural", "name": "Diego Italian (FREE)", "description": "🇮🇹 Natural Italian male - Unlimited", "provider": "edge", "cost_per_1000_chars": 0.0},
        
        # 🆓 FREE TIER - Azure Speech (500k chars/month)
        {"id": "azure:en-US-AriaNeural", "name": "Aria Azure (FREE)", "description": "🆓 High-quality neural - 500k/month", "provider": "azure", "cost_per_1000_chars": 0.0},
        {"id": "azure:en-US-JennyNeural", "name": "Jenny Azure (FREE)", "description": "🆓 Natural female - 500k/month", "provider": "azure", "cost_per_1000_chars": 0.0},
        {"id": "azure:en-US-GuyNeural", "name": "Guy Azure (FREE)", "description": "🆓 Professional male - 500k/month", "provider": "azure", "cost_per_1000_chars": 0.0},
        
        # 🇮🇹 ITALIAN VOICES - Azure Speech (FREE TIER)
        {"id": "azure:it-IT-ElsaNeural", "name": "Elsa Italian Azure (FREE)", "description": "🇮🇹 Premium Italian female - 500k/month", "provider": "azure", "cost_per_1000_chars": 0.0},
        {"id": "azure:it-IT-IsabellaNeural", "name": "Isabella Italian Azure (FREE)", "description": "🇮🇹 Professional Italian female - 500k/month", "provider": "azure", "cost_per_1000_chars": 0.0},
        {"id": "azure:it-IT-DiegoNeural", "name": "Diego Italian Azure (FREE)", "description": "🇮🇹 Natural Italian male - 500k/month", "provider": "azure", "cost_per_1000_chars": 0.0},
        
        # 💰 PREMIUM - ElevenLabs (High quality, expensive)
        {"id": "elevenlabs:9BWtsMINqrJLrRacOk9x", "name": "Aria (ElevenLabs)", "description": "💰 Premium quality", "provider": "elevenlabs", "cost_per_1000_chars": 0.33},
        {"id": "elevenlabs:zcAOhNBS3c14rBihAFp1", "name": "Adam (ElevenLabs)", "description": "💰 Deep, professional", "provider": "elevenlabs", "cost_per_1000_chars": 0.33},
        {"id": "elevenlabs:pNInz6obpgDQGcFmaJgB", "name": "Antoni (ElevenLabs)", "description": "💰 Warm, engaging", "provider": "elevenlabs", "cost_per_1000_chars": 0.33}
    ]
    
    return JSONResponse(content={
        "voices": voices,
        "recommendations": {
            "most_affordable": "openai:alloy",
            "completely_free": "edge:en-US-AriaNeural", 
            "best_quality_free": "azure:en-US-AriaNeural",
            "premium": "elevenlabs:9BWtsMINqrJLrRacOk9x"
        }
    })

# Health check endpoint
@app.get("/health")
async def health_check():
    return JSONResponse(content={
        "status": "healthy",
        "vector_store_ready": global_vector_store is not None,
        "chat_history_count": len(chat_history),
        "uptime": time.time()
    })

# Get document stats
@app.get("/documents/stats")
async def get_document_stats():
    if global_vector_store is None:
        return JSONResponse(content={"error": "No documents uploaded"}, status_code=400)
    
    try:
        # Get basic stats about uploaded documents
        uploaded_files = os.listdir(UPLOAD_FOLDER)
        total_size = sum(os.path.getsize(os.path.join(UPLOAD_FOLDER, f)) for f in uploaded_files)
        
        return JSONResponse(content={
            "document_count": len(uploaded_files),
            "total_size_kb": round(total_size / 1024, 2),
            "files": uploaded_files,
            "vector_store_ready": True
        })
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/test-ai")
async def test_ai_response():
    """Simple test endpoint to verify AI is working"""
    try:
        # Test with Ollama (free) first
        response = await get_ai_response_free("Hello, this is a test. Please respond briefly.", "ollama", "llama3.2:3b")
        return JSONResponse(content={"status": "success", "response": response, "provider": "ollama"})
    except Exception as ollama_error:
        try:
            # Fallback to OpenAI if Ollama fails
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
            response = llm.invoke("Hello, this is a test. Please respond briefly.").content
            return JSONResponse(content={"status": "success", "response": response, "provider": "openai"})
        except Exception as openai_error:
            return JSONResponse(content={
                "status": "error", 
                "ollama_error": str(ollama_error),
                "openai_error": str(openai_error)
            }, status_code=500)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7002))
    uvicorn.run(app, host="0.0.0.0", port=port)
