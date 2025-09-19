#!/usr/bin/env python3
"""
Veloci AI Chatbot - Interactive Conversation Mode
Startup script for testing the new conversational features
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'openai',
        'langchain',
        'elevenlabs',
        'python-dotenv'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Install them with: pip install -r requirements.txt")
        return False
    
    return True

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting Veloci AI Chatbot...")
    print("=" * 50)
    
    if not check_requirements():
        return
    
    # Check if .env file exists
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found!")
        print("📝 Create .env file with:")
        print("   OPENAI_API_KEY=your_openai_key")
        print("   ELEVENLABS_API_KEY=your_elevenlabs_key")
        return
    
    try:
        print("🔧 Starting server on http://localhost:7000")
        print("💬 Regular Chat: http://localhost:7000")
        print("🎤 Conversation Mode: http://localhost:7000/conversation")
        print("📱 Classic Interface: http://localhost:7000/classic")
        print("-" * 50)
        print("📡 WebSocket endpoints:")
        print("   💬 Chat: ws://localhost:7000/ws")
        print("   🗣️  Conversation: ws://localhost:7000/conversation")
        print("-" * 50)
        
        # Start the server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app:app", 
            "--host", "0.0.0.0", 
            "--port", "7000", 
            "--reload"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down server...")
    except Exception as e:
        print(f"❌ Server error: {e}")

def open_browser():
    """Open browser to the conversation mode"""
    print("🌐 Opening conversation mode in browser...")
    try:
        webbrowser.open('http://localhost:7000/conversation')
    except Exception as e:
        print(f"❌ Could not open browser: {e}")

if __name__ == "__main__":
    print("🎤 Veloci AI - Interactive Conversation Mode")
    print("✨ Natural voice conversations like iPhone ChatGPT")
    print()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--browser":
        # Wait a moment for server to start, then open browser
        import threading
        timer = threading.Timer(2.0, open_browser)
        timer.start()
    
    start_server()