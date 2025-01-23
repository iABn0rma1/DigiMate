import gradio as gr
import google.generativeai as genai
from gtts import gTTS
import speech_recognition as sr
import pyttsx3
import os
from pathlib import Path
import json
from datetime import datetime
from dotenv import load_dotenv
import torch
from TTS.api import TTS
import numpy as np
from scipy.io import wavfile
import tempfile

# Load the .env file
load_dotenv()

# Configuration
gemini_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel('gemini-pro')

# Initialize speech recognition
recognizer = sr.Recognizer()

# Initialize Coqui TTS with Glow-TTS model
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/en/ljspeech/glow-tts").to(device)

# Directory setup
responses_dir = Path("responses")
responses_dir.mkdir(exist_ok=True)

chat_history_dir = Path("chat_history")
chat_history_dir.mkdir(exist_ok=True)

BUDDY_NAME = "Buddy"

class EmotionalSpeech:
    def __init__(self):
        self.tts = tts
        self.emotions = {
            "happy": {"speed": 1.2},
            "sad": {"speed": 0.8},
            "excited": {"speed": 1.3},
            "calm": {"speed": 0.9},
            "singing": {"speed": 1.1},
            "storytelling": {"speed": 0.95}
        }
    
    def detect_emotion(self, text):
        # Simple keyword-based emotion detection
        keywords = {
            "happy": ["yay", "wonderful", "happy", "joy", "excited", "great"],
            "sad": ["sad", "sorry", "unfortunate", "miss", "difficult"],
            "excited": ["wow", "amazing", "awesome", "incredible", "fantastic"],
            "calm": ["gentle", "peaceful", "quiet", "soft"],
            "singing": ["🎵", "sing", "song", "la la", "tune"],
            "storytelling": ["once upon a time", "story", "tale", "chapter"]
        }
        
        text_lower = text.lower()
        for emotion, words in keywords.items():
            if any(word in text_lower for word in words):
                return emotion
        return "calm"  # default emotion

    def preprocess_text(self, text):
        # Ensure minimum text length
        if len(text.strip()) < 3:
            text = text + " " * (3 - len(text.strip()))
        return text

    def synthesize(self, text, emotion=None):
        if emotion is None:
            emotion = self.detect_emotion(text)
            
        params = self.emotions.get(emotion, self.emotions["calm"])
        
        # Preprocess text to handle short inputs
        text = self.preprocess_text(text)
        
        try:
            # First try with Glow-TTS
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                self.tts.tts_to_file(
                    text=text,
                    file_path=temp_file.name,
                    speed=params["speed"]
                )
                return temp_file.name
        except Exception as e:
            print(f"Glow-TTS error: {e}")
            # Fall back to gTTS for very short texts or if Glow-TTS fails
            try:
                temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                tts_fallback = gTTS(text=text, lang='en', tld='com', slow=False)
                tts_fallback.save(temp_file.name)
                return temp_file.name
            except Exception as e:
                print(f"gTTS error: {e}")
                return None

class BuddyBear:
    def __init__(self):
        self.user_name = None
        self.speech_synthesizer = EmotionalSpeech()

    def format_response(self, user_input, history):
        recent_history = history[-4:] if history else []
        history_context = "\n".join([
            f"{'Child' if msg['role'] == 'user' else BUDDY_NAME}: {msg['content']}"
            for msg in recent_history
        ])

        context = f"""
        You are Buddy, a friendly voice assistant for children aged 4-10.
        Current user's name: {self.user_name or 'unknown'}.
        
        Please ensure your responses are at least a few words long to help with speech synthesis.

        Recent conversation:
        {history_context}

        Child's message: {user_input}

        Continue the conversation naturally. Use emojis to indicate emotions and tone:
        - 😊 for happy responses
        - 🎵 for singing
        - 📖 for storytelling
        - 😃 for excited responses
        - 😢 for empathetic/sad responses
        - 😌 for calm/soothing responses

        Make learning fun, and ensure responses are age-appropriate and friendly.
        """

        response = model.generate_content(context)
        return response.text

    def generate_voice(self, text):
        try:
            # Add some padding for very short responses
            if len(text.strip()) < 3:
                text = f"{text.strip()} . . ."
                
            # Detect if the text contains special content types
            if "🎵" in text:
                return self.speech_synthesizer.synthesize(text, "singing")
            elif "📖" in text:
                return self.speech_synthesizer.synthesize(text, "storytelling")
            else:
                # Let the EmotionalSpeech class detect the emotion
                return self.speech_synthesizer.synthesize(text)
        except Exception as e:
            print(f"TTS error: {e}")
            # Final fallback to pyttsx3
            try:
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                engine = pyttsx3.init()
                engine.save_to_file(text, temp_file.name)
                engine.runAndWait()
                return temp_file.name
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
            return None

class ChatInterface:
    def __init__(self):
        self.buddy = BuddyBear()

    def save_chat_history(self, chat_history):
        if not self.buddy.user_name:
            return
        filename = chat_history_dir / f"{self.buddy.user_name}_chat.json"
        with open(filename, "w") as f:
            json.dump(chat_history, f)

    def load_chat_history(self):
        if not self.buddy.user_name:
            return []
        filename = chat_history_dir / f"{self.buddy.user_name}_chat.json"
        if filename.exists():
            with open(filename) as f:
                return json.load(f)
        return []

    def process_message(self, message, history):
        if not self.buddy.user_name:
            self.buddy.user_name = message
            greeting = f"Hi {self.buddy.user_name}! It's great to meet you! How was your day?"
            audio_path = self.buddy.generate_voice(greeting)
            return "", audio_path, [{"role": "assistant", "content": greeting}]

        history = history or self.load_chat_history()
        response = self.buddy.format_response(message, history)
        audio_path = self.buddy.generate_voice(response)

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        self.save_chat_history(history)

        return "", audio_path, history

    def process_voice_message(self, audio_file, history):
        transcribed_text = self.buddy.transcribe_audio(audio_file)
        if not transcribed_text:
            return "I couldn't hear that clearly. Could you try saying that again?", None, history

        return self.process_message(transcribed_text, history)

    def create_interface(self):
        with gr.Blocks(theme=gr.themes.Soft()) as interface:
            gr.Markdown("# Hi! I'm Buddy! 🐻")
            gr.Markdown("Let's talk! I love making new friends and learning new things!")

            with gr.Row():
                username = gr.Textbox(
                    label="Tell me your name!", placeholder="Your name", scale=2
                )

            chatbot = gr.Chatbot(
                label="Our Chat", bubble_full_width=False, height=400, type="messages"
            )

            with gr.Row():
                msg = gr.Textbox(
                    label="Send me a message!", placeholder="What's on your mind?", scale=3
                )
                audio_msg = gr.Audio(
                    label="Or talk to me!", sources=["microphone"], type="filepath"
                )
                clear = gr.Button("Start Over")

            audio_output = gr.Audio(label="Listen to my response!")

            msg.submit(
                self.process_message,
                inputs=[msg, chatbot],
                outputs=[msg, audio_output, chatbot]
            )

            audio_msg.stop_recording(
                self.process_voice_message,
                inputs=[audio_msg, chatbot],
                outputs=[msg, audio_output, chatbot]
            )

            def clear_chat():
                self.buddy.user_name = None
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
