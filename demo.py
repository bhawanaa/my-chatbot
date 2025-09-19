#!/usr/bin/env python3
"""
Veloci AI Interactive Demo Script
Demonstrates the new interactive features of the chatbot
"""

import asyncio
import aiohttp
import json
import time
import os

class VelociAIDemo:
    def __init__(self, base_url="http://localhost:7000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_health(self):
        """Check if the server is running"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                data = await response.json()
                print("🟢 Server Status:", data)
                return True
        except Exception as e:
            print("🔴 Server not available:", e)
            return False
    
    async def upload_demo_document(self):
        """Upload a sample document"""
        # Create a simple demo document
        demo_content = """
        Welcome to Veloci AI Interactive Chatbot!
        
        This is a demonstration document that showcases our new interactive features:
        
        1. Real-time WebSocket communication
        2. Voice input and output capabilities
        3. Advanced document processing
        4. Smart question suggestions
        5. Beautiful modern interface
        
        Key Features:
        - Multi-format document support (PDF, DOCX, XLSX, TXT, Images)
        - OCR text extraction from images
        - Voice-to-text and text-to-voice conversion
        - Context-aware AI responses
        - Chat history management
        - Document summarization
        
        Try asking questions like:
        - "What are the key features?"
        - "How does voice input work?"
        - "What file formats are supported?"
        - "Summarize this document"
        """
        
        # Write demo document
        demo_file = "demo_document.txt"
        with open(demo_file, "w") as f:
            f.write(demo_content)
        
        # Upload the document
        try:
            with open(demo_file, "rb") as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename=demo_file)
                
                async with self.session.post(f"{self.base_url}/upload", data=data) as response:
                    result = await response.json()
                    print("📄 Document Upload:", result.get("message"))
                    print("📊 Stats:", result.get("stats"))
                    
            # Clean up
            os.remove(demo_file)
            return True
            
        except Exception as e:
            print("❌ Upload failed:", e)
            return False
    
    async def demo_question_answering(self):
        """Demonstrate Q&A functionality"""
        questions = [
            "What are the key features of this chatbot?",
            "How does voice input work?",
            "What file formats are supported?",
            "What makes this chatbot interactive?"
        ]
        
        print("\n🤖 Q&A Demonstration:")
        print("=" * 50)
        
        for i, question in enumerate(questions, 1):
            print(f"\n❓ Question {i}: {question}")
            
            try:
                data = aiohttp.FormData()
                data.add_field('question', question)
                
                async with self.session.post(f"{self.base_url}/ask", data=data) as response:
                    result = await response.json()
                    print(f"🤖 Answer: {result.get('answer')}")
                    
                    # Small delay for demo purposes
                    await asyncio.sleep(1)
                    
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def demo_suggestions(self):
        """Demonstrate suggestion functionality"""
        print("\n💡 Smart Suggestions Demo:")
        print("=" * 50)
        
        sample_answer = "The chatbot supports voice input, document processing, and real-time communication."
        
        try:
            data = aiohttp.FormData()
            data.add_field('last_answer', sample_answer)
            
            async with self.session.post(f"{self.base_url}/suggest", data=data) as response:
                result = await response.json()
                suggestions = result.get('suggestions', [])
                
                print("🔮 Based on the answer, here are suggested follow-up questions:")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"   {i}. {suggestion}")
                    
        except Exception as e:
            print(f"❌ Suggestion error: {e}")
    
    async def demo_document_stats(self):
        """Show document statistics"""
        print("\n📊 Document Statistics:")
        print("=" * 50)
        
        try:
            async with self.session.get(f"{self.base_url}/documents/stats") as response:
                result = await response.json()
                print(f"📁 Documents: {result.get('document_count', 0)}")
                print(f"💾 Total Size: {result.get('total_size_kb', 0)} KB")
                print(f"📋 Files: {', '.join(result.get('files', []))}")
                
        except Exception as e:
            print(f"❌ Stats error: {e}")
    
    async def demo_summarization(self):
        """Demonstrate document summarization"""
        print("\n📝 Document Summarization Demo:")
        print("=" * 50)
        
        try:
            async with self.session.post(f"{self.base_url}/summarize") as response:
                result = await response.json()
                summary = result.get('summary', 'No summary available')
                print(f"📄 Summary: {summary}")
                
        except Exception as e:
            print(f"❌ Summarization error: {e}")
    
    async def get_available_voices(self):
        """Show available TTS voices"""
        print("\n🎤 Available Voices:")
        print("=" * 50)
        
        try:
            async with self.session.get(f"{self.base_url}/voices") as response:
                result = await response.json()
                voices = result.get('voices', [])
                
                for voice in voices:
                    print(f"🔊 {voice['name']}: {voice['description']}")
                    
        except Exception as e:
            print(f"❌ Voice list error: {e}")

async def main():
    """Run the interactive demo"""
    print("🚀 Veloci AI Interactive Demo")
    print("=" * 50)
    print("This demo showcases the new interactive features")
    print("Make sure the server is running at http://localhost:7000")
    print()
    
    async with VelociAIDemo() as demo:
        # Check if server is running
        if not await demo.check_health():
            print("❌ Please start the server first:")
            print("   python start_chatbot.py")
            return
        
        print("\n🎯 Starting Interactive Demo...")
        
        # Run demo steps
        await demo.get_available_voices()
        await demo.upload_demo_document()
        await demo.demo_document_stats()
        await demo.demo_question_answering()
        await demo.demo_suggestions()
        await demo.demo_summarization()
        
        print("\n✅ Demo completed!")
        print("🌐 Open http://localhost:7000 to try the interactive interface")
        print("🎙️ Features to try:")
        print("   • Upload documents via drag & drop")
        print("   • Use voice input (microphone button)")
        print("   • Adjust voice settings in sidebar")
        print("   • Click suggestion chips")
        print("   • Try the document summarization button")

if __name__ == "__main__":
    asyncio.run(main())
