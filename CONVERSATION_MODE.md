# 🎤 Veloci AI - Interactive Conversation Mode

## 🚀 What's New - iPhone-Style Voice Conversations!

Your chatbot now features a **conversational mode** just like ChatGPT on iPhone! Have natural, flowing conversations with AI using your voice - no more typing required.

## ✨ Key Features

### 🗣️ **Natural Voice Conversations**
- **Hold and speak** - Just like iPhone ChatGPT voice mode
- **Context awareness** - AI remembers your conversation history
- **Intelligent responses** - Natural, conversational AI responses
- **Real-time feedback** - See transcripts and status updates

### 🎵 **Full Voice Control**
- **5 AI voices** - Choose your preferred personality
- **Speed adjustment** - 0.5x to 2.0x playback speed
- **High-quality TTS** - Crystal clear AI speech
- **Voice switching** - Change voices mid-conversation

### 💬 **Smart Conversation Flow**
- **WebSocket real-time** - Instant communication
- **Session management** - Maintains conversation context
- **Document integration** - AI can reference uploaded documents
- **Error handling** - Graceful recovery from issues

## 🎯 How to Use

### 1. **Start Conversation Mode**
```bash
# Option 1: Use the startup script
python start_conversation.py

# Option 2: Start server manually
python -m uvicorn app:app --host 0.0.0.0 --port 7000 --reload
```

### 2. **Access the Interface**
- **Conversation Mode**: http://localhost:7000/conversation
- **Regular Chat**: http://localhost:7000
- **Classic Interface**: http://localhost:7000/classic

### 3. **Voice Conversation Steps**
1. 📱 **Open** http://localhost:7000/conversation
2. 🎤 **Hold** the large microphone button
3. 🗣️ **Speak** naturally - say anything!
4. 👂 **Listen** to AI response in your chosen voice
5. 🔄 **Continue** the natural conversation flow

## 🛠️ Technical Implementation

### **Backend Architecture**
```python
# New WebSocket endpoint for conversations
@app.websocket("/conversation")
async def conversation_websocket(websocket: WebSocket):
    # Real-time voice conversation handling
    
# Conversation session management
class ConversationSession:
    def __init__(self, session_id: str):
        self.context = []  # Conversation history
        self.voice_settings = {}  # User preferences
        self.is_active = False
```

### **Frontend Features**
- **Large voice button** with visual feedback
- **Real-time status updates** showing conversation state
- **Voice controls** for personalization
- **Conversation transcript** showing message history
- **Visual indicators** for listening, processing, speaking

### **WebSocket Messages**
```javascript
// User voice input
{
    "type": "voice_input",
    "content": "Hello, how are you?",
    "voice_settings": {
        "voice_id": "9BWtsMINqrJLrRacOk9x",
        "speed": 1.0
    }
}

// AI response
{
    "type": "ai_response", 
    "content": "I'm doing great! How can I help you?",
    "audio_url": "/static/output.mp3",
    "session_id": "conv_1234567890"
}
```

## 🎨 Interface Design

### **Visual Elements**
- **Dark gradient background** - Modern, professional look
- **Large circular voice button** - Easy to use, visual feedback
- **Animated ripples** - Shows listening state
- **Real-time transcript** - See conversation history
- **Status indicators** - Clear feedback on current state

### **Button States**
- 🎤 **Ready** - Purple gradient, microphone icon
- 👂 **Listening** - Green gradient, pulsing animation
- 🔄 **Processing** - Orange gradient, spinning icon
- 🔊 **Speaking** - Visual feedback during AI response

## 🎵 Available Voices

| Voice | Gender | Personality | Voice ID |
|-------|---------|------------|----------|
| **Aria** | Female | Young, friendly | `9BWtsMINqrJLrRacOk9x` |
| **Adam** | Male | Deep, professional | `zcAOhNBS3c14rBihAFp1` |
| **Antoni** | Male | Warm, engaging | `pNInz6obpgDQGcFmaJgB` |
| **Arnold** | Male | Authoritative | `VR6AewLTigWG4xSOukaG` |
| **Antoni 2** | Male | Smooth, articulate | `ErXwobaYiN019PkySvjV` |

## 📱 Mobile Experience

### **Responsive Design**
- **Touch-friendly** large buttons
- **Mobile-optimized** layouts
- **Swipe gestures** for navigation
- **Voice-first** interface design

### **iOS/Android Compatibility**
- **Web Speech API** support
- **Touch event handling** 
- **Audio autoplay** management
- **Progressive enhancement**

## 🔧 Configuration Options

### **Environment Variables**
```env
# Required API keys
OPENAI_API_KEY=sk-your_openai_key
ELEVENLABS_API_KEY=your_elevenlabs_key

# Optional settings  
CONVERSATION_TIMEOUT=300  # 5 minutes session timeout
DEFAULT_VOICE_ID=9BWtsMINqrJLrRacOk9x  # Default to Aria
DEFAULT_SPEED=1.0  # Default playback speed
```

### **Voice Settings**
- **Speed range**: 0.5x to 2.0x
- **Voice switching**: Real-time during conversation
- **Quality**: High-fidelity neural TTS
- **Language**: English (extensible to other languages)

## 🚀 Getting Started

### **Quick Start**
1. **Clone and setup** your chatbot project
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Add API keys** to `.env` file
4. **Start conversation mode**: `python start_conversation.py`
5. **Open browser**: http://localhost:7000/conversation
6. **Start talking!** 🎤

### **Upload Documents (Optional)**
1. Go to regular chat mode: http://localhost:7000
2. Upload PDF, DOCX, or other documents
3. Return to conversation mode
4. Ask questions about your documents via voice!

## 💡 Usage Tips

### **Best Practices**
- 🎤 **Clear speech** - Speak clearly and at normal pace
- 📱 **Good microphone** - Use quality audio input device
- 🌐 **Chrome/Edge** - Best browser support for Web Speech API
- 🔊 **Enable audio** - Allow audio autoplay for best experience

### **Conversation Flow**
- **Natural speech** - Talk like you would to a human
- **Context building** - AI remembers previous messages
- **Document queries** - Ask about uploaded files
- **Voice switching** - Change AI voice anytime during chat

### **Troubleshooting**
- **Microphone permissions** - Allow mic access when prompted
- **Audio playback** - Click page if audio is blocked
- **WebSocket errors** - Refresh page to reconnect
- **Speech recognition** - Ensure quiet environment

## 🔍 Advanced Features

### **Context Management**
- **Session tracking** - Maintains conversation state
- **History retention** - Keeps last 10 message pairs
- **Smart timeout** - 5-minute inactivity cleanup
- **Document integration** - References uploaded content

### **Real-time Communication**
- **WebSocket protocol** - Low-latency messaging
- **Status updates** - Real-time conversation state
- **Error recovery** - Automatic reconnection
- **Performance optimization** - Efficient message handling

### **Accessibility**
- **Keyboard navigation** - Full keyboard support
- **Screen reader friendly** - Proper ARIA labels
- **High contrast** - Readable in all lighting
- **Large touch targets** - Easy mobile interaction

## 🎯 Comparison with iPhone ChatGPT

| Feature | iPhone ChatGPT | Veloci AI Conversation |
|---------|----------------|----------------------|
| **Voice Input** | ✅ Hold to speak | ✅ Hold to speak |
| **AI Voices** | ✅ Multiple voices | ✅ 5 voice options |
| **Real-time** | ✅ Live conversation | ✅ WebSocket real-time |
| **Context** | ✅ Remembers chat | ✅ Session management |
| **Documents** | ❌ Limited | ✅ Full document support |
| **Customization** | ❌ Limited | ✅ Voice & speed control |
| **Web Access** | ❌ App only | ✅ Any browser |

## 📈 Future Enhancements

### **Planned Features**
- 🌍 **Multi-language support** - Spanish, French, etc.
- 🎵 **Voice cloning** - Custom voice creation
- 📞 **Phone integration** - Call-in voice interface
- 🤖 **AI interruption** - Stop AI mid-sentence
- 🎭 **Personality modes** - Different conversation styles

### **Technical Roadmap**
- **Voice activity detection** - Automatic speech triggering
- **Noise cancellation** - Better audio processing
- **Streaming responses** - Real-time AI speech generation
- **Offline mode** - Local speech processing
- **API integrations** - Connect external services

## 🎊 Success! 

Your chatbot now has **iPhone-style voice conversations**! 

🎤 **Natural speech** ↔️ 🤖 **AI responses** ↔️ 🔊 **Voice playback**

The conversation flows naturally with context awareness, document integration, and full voice customization - just like talking to a real person!

---

**Enjoy your new conversational AI experience!** 🚀✨