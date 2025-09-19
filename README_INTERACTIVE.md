# Veloci AI - Interactive Chatbot 🤖

An enhanced, interactive AI chatbot with voice capabilities, real-time communication, and document processing features.

## ✨ New Interactive Features

### 🎙️ Voice Integration
- **Speech Recognition**: Press and hold the microphone button to speak
- **Text-to-Speech**: AI responses are automatically converted to speech
- **Multiple Voices**: Choose from different AI voices (Aria, Adam, Antoni)
- **Speed Control**: Adjust playback speed (0.5x to 2.0x)

### 💬 Enhanced Chat Interface
- **Real-time Communication**: WebSocket-based instant messaging
- **Modern UI**: Beautiful gradient design with animations
- **Typing Indicators**: See when AI is processing your request
- **Message Timestamps**: Track conversation history
- **Smart Suggestions**: Get follow-up question recommendations

### 📁 Advanced Document Processing
- **Drag & Drop Upload**: Simply drag files into the upload area
- **Multiple File Support**: PDF, DOCX, XLSX, TXT, images (JPG, PNG, TIF)
- **Document Statistics**: View file count, total size, and processing info
- **Auto-Summarization**: Get instant summaries of uploaded documents
- **Image Text Extraction**: OCR processing for image-based documents

### 🔄 Real-time Features
- **Streaming Responses**: See AI responses as they're generated
- **Background Processing**: Non-blocking file uploads and processing
- **Health Monitoring**: System status and performance indicators
- **Chat History**: Persistent conversation storage and retrieval

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables
Create a `.env` file with:
```env
OPENAI_API_KEY=your_openai_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
TESSERACT_CMD=/usr/bin/tesseract  # Optional: for OCR
```

### 3. Start the Application
```bash
# Using the startup script (recommended)
python start_chatbot.py

# Or directly
python app.py
```

### 4. Access the Interface
- **Interactive Interface**: http://localhost:7000
- **Classic Interface**: http://localhost:7000/classic
- **Health Check**: http://localhost:7000/health

## 🎯 Key Endpoints

### WebSocket Communication
- `ws://localhost:7000/ws` - Real-time chat communication

### REST API Endpoints
- `POST /ask` - Send questions and get AI responses
- `POST /ask/stream` - Streaming response endpoint
- `POST /upload` - Upload and process documents
- `POST /transcribe` - Audio-to-text transcription
- `POST /tts` - Custom text-to-speech generation
- `POST /suggest` - Get follow-up question suggestions
- `POST /summarize` - Generate document summaries
- `GET /chat/history` - Retrieve chat history
- `POST /chat/clear` - Clear chat history
- `GET /voices` - Get available TTS voices
- `GET /documents/stats` - Document processing statistics

## 💡 Usage Tips

### Voice Features
1. **Speech Input**: Click and hold the microphone button while speaking
2. **Voice Selection**: Choose your preferred AI voice from the sidebar
3. **Speed Adjustment**: Use the speed slider for comfortable listening

### Document Processing
1. **Upload Methods**: 
   - Drag and drop files onto the upload area
   - Click the upload area to select files
   - Multiple files can be uploaded simultaneously

2. **Supported Formats**:
   - **Text**: PDF, DOCX, TXT
   - **Data**: XLSX spreadsheets
   - **Images**: JPG, JPEG, PNG, TIF (with OCR)

3. **Smart Features**:
   - Automatic text extraction from images
   - PDF page-by-page processing
   - Excel sheet analysis
   - Document summarization

### Chat Interface
1. **Message Input**: 
   - Type messages in the text area
   - Press Enter to send (Shift+Enter for new line)
   - Use voice input for hands-free operation

2. **Interactive Elements**:
   - Click suggestion chips for quick questions
   - View real-time typing indicators
   - See message timestamps
   - Visual feedback for all actions

## 🛠️ Technical Architecture

### Backend Components
- **FastAPI**: High-performance web framework
- **LangChain**: Document processing and AI orchestration
- **OpenAI GPT**: Language model for responses
- **ElevenLabs**: Text-to-speech generation
- **FAISS**: Vector storage for document similarity search
- **WebSockets**: Real-time communication

### Frontend Features
- **Bootstrap 5**: Responsive UI framework
- **Font Awesome**: Icon library
- **Web Speech API**: Browser-based speech recognition
- **HTML5 Audio**: Audio playback capabilities
- **WebSocket Client**: Real-time communication
- **Drag & Drop API**: File upload interface

### Processing Pipeline
1. **Document Upload** → Text Extraction → Chunking → Vector Embedding → Storage
2. **User Question** → Context Retrieval → AI Processing → Response Generation → TTS → Delivery
3. **Voice Input** → Speech Recognition → Text Processing → Response Generation

## 🔧 Configuration Options

### Environment Variables
```env
# Required
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=...

# Optional
TESSERACT_CMD=/usr/bin/tesseract  # For OCR processing
```

### Voice Settings
- **Available Voices**: Aria, Adam, Antoni, Arnold
- **Speed Range**: 0.5x to 2.0x playback speed
- **Quality**: High-quality neural TTS

### Upload Limits
- **File Types**: PDF, DOCX, XLSX, TXT, JPG, JPEG, PNG, TIF
- **Processing**: Automatic text extraction and chunking
- **Storage**: Local filesystem with vector database

## 🎨 Customization

### UI Themes
The interface uses CSS custom properties for easy theming:
```css
:root {
  --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  --chat-bubble-user: linear-gradient(45deg, #007bff, #6f42c1);
  --chat-bubble-ai: #e9ecef;
}
```

### Voice Customization
Add new voices by updating the voice selection in `index_interactive.html`:
```javascript
const voices = [
    {"id": "voice_id", "name": "Voice Name", "description": "Description"}
];
```

## 🔍 Troubleshooting

### Common Issues
1. **Audio Not Playing**: Click anywhere on the page to enable audio autoplay
2. **Speech Recognition Not Working**: Ensure microphone permissions are granted
3. **File Upload Fails**: Check file format and size limitations
4. **TTS Not Working**: Verify ElevenLabs API key is valid

### Debug Mode
Enable detailed logging by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Optimization

### Recommendations
1. **Vector Store**: Consider using persistent storage for large document collections
2. **Caching**: Implement Redis for response caching
3. **Load Balancing**: Use multiple instances for high traffic
4. **CDN**: Serve static assets from a CDN

### Monitoring
- Health check endpoint: `/health`
- Document statistics: `/documents/stats`
- Chat history management: Built-in cleanup

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- ElevenLabs for TTS technology
- LangChain for document processing
- FastAPI for the web framework
- The open-source community for various libraries

---

**Made with ❤️ for interactive AI conversations**
