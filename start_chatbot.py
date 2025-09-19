#!/usr/bin/env python3
"""
Enhanced Interactive Chatbot Startup Script
"""
import os
import sys
import subprocess
import webbrowser
import time

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import fastapi
        import openai
        import langchain
        import aiohttp
        print("✅ All core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Please install requirements: pip install -r requirements.txt")
        return False

def start_server():
    """Start the FastAPI server"""
    try:
        from app import app
        import uvicorn
        
        print("🚀 Starting Veloci AI Interactive Chatbot...")
        print("📍 Server will be available at: http://localhost:7000")
        print("🔗 Interactive Interface: http://localhost:7000")
        print("🔗 Classic Interface: http://localhost:7000/classic")
        print("📊 Health Check: http://localhost:7000/health")
        print("\n" + "="*50)
        
        # Start the server
        uvicorn.run(app, host="0.0.0.0", port=7000, log_level="info")
        
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

def main():
    """Main startup function"""
    print("🤖 Veloci AI - Interactive Chatbot")
    print("="*40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment")
        print("   Please add it to your .env file")
    
    if not os.getenv("ELEVENLABS_API_KEY"):
        print("⚠️  Warning: ELEVENLABS_API_KEY not found in environment")
        print("   TTS features will not work without this key")
    
    print("\n🎯 Features Available:")
    print("   • Interactive Chat Interface")
    print("   • Real-time WebSocket Communication")
    print("   • Voice Input (Speech Recognition)")
    print("   • Text-to-Speech Output")
    print("   • Document Upload & Processing")
    print("   • Smart Question Suggestions")
    print("   • Chat History Management")
    print("   • Multiple Voice Options")
    print("   • Document Summarization")
    print("   • Streaming Responses")
    
    print("\n📋 Usage Tips:")
    print("   • Upload documents first for context-aware responses")
    print("   • Use voice input by holding the microphone button")
    print("   • Adjust voice speed and selection in the sidebar")
    print("   • Try the suggested follow-up questions")
    print("   • Drag and drop files for easy upload")
    
    # Start the server
    start_server()

if __name__ == "__main__":
    main()
