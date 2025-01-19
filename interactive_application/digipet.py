import gradio as gr
import google.generativeai as genai
from gtts import gTTS
import speech_recognition as sr
import pyttsx3
import os
from pathlib import Path
import json
import random
from datetime import datetime
from dotenv import load_dotenv

# Load the.env file
load_dotenv()

# Configuration
gemini_key = os.getenv('GEMINI_API_KEY')  # Replace with your actual API key

# Initialize APIs
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

# Initialize speech recognition
recognizer = sr.Recognizer()

# Initialize pyttsx3 for offline TTS fallback
engine = pyttsx3.init()
# Configure voice to be more kid-friendly
voices = engine.getProperty('voices')
# Try to find a higher-pitched voice
for voice in voices:
    if "female" in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break
engine.setProperty('rate', 150)  # Slower speed
engine.setProperty('volume', 0.9)

# [Previous character configuration and guidelines remain the same]
BUDDY_NAME = "Buddy"
MAX_RESPONSE_LENGTH = 200

# Safety guidelines for internal use
SAFETY_GUIDELINES = """
You are Buddy, a warm and friendly companion who loves talking with children. You should:
1. Keep conversations natural and age-appropriate (ages 4-10)
2. Show genuine interest in what the child says
3. Be encouraging but not overly enthusiastic
4. Use simple language and short sentences
5. Incorporate gentle learning moments naturally
6. Be patient and understanding
7. Never share personal information
8. Redirect sensitive topics to parents/teachers
9. Keep responses concise and clear
10. Maintain a warm, caring tone

Important conversation guidelines:
- Speak naturally like a friend, not a teacher
- Let the conversation flow organically
- Ask follow-up questions about things the child mentions
- Share relevant stories and experiences
- Show empathy and understanding
- Make learning fun through casual conversation
"""

class BuddyBear:
    def __init__(self):
        self.chat_history = {}
        
    def format_response(self, user_input, username, history):
        # Include recent chat history for context
        recent_history = history[-4:] if history else []
        history_context = "\n".join([
            f"{'Child' if msg['role'] == 'user' else 'Buddy'}: {msg['content']}"
            for msg in recent_history
        ])
        
        context = f"""
        You are having a natural conversation with {username}, a child.
        
        Recent conversation:
        {history_context}
        
        Child's message: {user_input}
        
        Follow these guidelines:
        {SAFETY_GUIDELINES}
        
        Respond naturally as Buddy, keeping the conversation flowing and engaging.
        """
        
        response = model.generate_content(context)
        return response.text

    def generate_voice(self, text, username):
        try:
            # First try gTTS (requires internet)
            audio_path = f"responses/{username}_latest.mp3"
            tts = gTTS(text=text, lang='en', tld='com', slow=False)
            tts.save(audio_path)
            return audio_path
        except Exception as e:
            print(f"gTTS error: {e}")
            try:
                # Fallback to pyttsx3 (offline)
                audio_path = f"responses/{username}_latest.mp3"
                engine.save_to_file(text, audio_path)
                engine.runAndWait()
                return audio_path
            except Exception as e:
                print(f"pyttsx3 error: {e}")
                return None

    def transcribe_audio(self, audio_file):
        try:
            with sr.AudioFile(audio_file) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
                return text
        except Exception as e:
            print(f"Speech recognition error: {e}")
            try:
                with sr.AudioFile(audio_file) as source:
                    audio = recognizer.record(source)
                    text = recognizer.recognize_sphinx(audio)
                    return text
            except Exception as e:
                print(f"Offline speech recognition error: {e}")
                return None

class ChatInterface:
    def __init__(self):
        self.buddy = BuddyBear()
        self.chat_dir = Path("chat_history")
        self.chat_dir.mkdir(exist_ok=True)
        
        if not os.path.exists("responses"):
            os.makedirs("responses")

    def save_chat_history(self, chat_history, username):
        filename = self.chat_dir / f"{username}_chat.json"
        with open(filename, "w") as f:
            json.dump(chat_history, f)

    def load_chat_history(self, username):
        filename = self.chat_dir / f"{username}_chat.json"
        if filename.exists():
            with open(filename) as f:
                return json.load(f)
        return []

    def process_message(self, message, history, username):
        if not username:
            return "Please tell me your name first!", None, history
        
        history = history or self.load_chat_history(username)
        response = self.buddy.format_response(message, username, history)
        audio_path = self.buddy.generate_voice(response, username)
        
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        self.save_chat_history(history, username)
        
        return "", audio_path, history

    def process_voice_message(self, audio_file, history, username):
        if not username:
            return "Please tell me your name first!", None, history
        
        transcribed_text = self.buddy.transcribe_audio(audio_file)
        if not transcribed_text:
            return "I couldn't hear that clearly. Could you try saying that again?", None, history
        
        return self.process_message(transcribed_text, history, username)

    def create_interface(self):
        with gr.Blocks(theme=gr.themes.Soft()) as interface:
            gr.Markdown("# Hi! I'm Buddy! 🐻")
            gr.Markdown("I love making new friends and having fun conversations!")
            
            with gr.Row():
                username = gr.Textbox(
                    label="First, tell me your name!",
                    placeholder="Your name",
                    scale=2
                )
            
            chatbot = gr.Chatbot(
                label="Our Chat",
                bubble_full_width=False,
                height=400,
                type="messages"
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Send me a message!",
                    placeholder="What's on your mind?",
                    scale=3
                )
                audio_msg = gr.Audio(
                    label="Or talk to me!",
                    sources=["microphone"],
                    type="filepath"
                )
                clear = gr.Button("Start Over")
            
            audio_output = gr.Audio(label="Listen to my response!")
            
            msg.submit(
                self.process_message,
                inputs=[msg, chatbot, username],
                outputs=[msg, audio_output, chatbot]
            )
            
            audio_msg.stop_recording(
                self.process_voice_message,
                inputs=[audio_msg, chatbot, username],
                outputs=[msg, audio_output, chatbot]
            )
            
            def clear_chat():
                return None, None, None
            
            clear.click(
                clear_chat,
                outputs=[msg, audio_output, chatbot]
            )
            
        return interface

def main():
    chat_interface = ChatInterface()
    demo = chat_interface.create_interface()
    demo.launch(share=True)

if __name__ == "__main__":
    main()
