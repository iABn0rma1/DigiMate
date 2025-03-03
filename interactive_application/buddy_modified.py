import warnings
import gradio as gr
import google.generativeai as genai
from gtts import gTTS
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
from bark import generate_audio, preload_models, SAMPLE_RATE
import soundfile as sf
import random
import whisper  # Import Whisper for STT

# Optional: Suppress the specific FutureWarning from torch.load in Bark
warnings.filterwarnings(
    "ignore", 
    message="You are using `torch.load` with `weights_only=False`"
)

# Load the .env file
load_dotenv()

# Configuration
gemini_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel('gemini-pro')

# Initialize Coqui TTS with Glow-TTS model
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/en/ljspeech/glow-tts").to(device)

# Preload Bark models
preload_models()

# Load OpenAI Whisper model (using the 'base' model; change as needed)
whisper_model = whisper.load_model("base")

# Directory setup
responses_dir = Path("responses")
responses_dir.mkdir(exist_ok=True)

chat_history_dir = Path("chat_history")
chat_history_dir.mkdir(exist_ok=True)

BUDDY_NAME = "Buddy"

class EmotionalSpeech:
    def __init__(self):
        self.emotions = {
            "happy": {"voice_preset": "v2/en_speaker_6", "speed": 1.2},
            "sad": {"voice_preset": "v2/en_speaker_3", "speed": 0.8},
            "excited": {"voice_preset": "v2/en_speaker_9", "speed": 1.3},
            "calm": {"voice_preset": "v2/en_speaker_2", "speed": 0.9},
            "singing": {"voice_preset": "v2/en_speaker_7", "speed": 1.1},
            "storytelling": {"voice_preset": "v2/en_speaker_4", "speed": 0.95}
        }
        
        self.fallback_explanations = [
            "Oops! My magic voice box is doing a little dance today! 🎭",
            "Wow, I just switched to my backup voice superhero mode! 🦸",
            "Listen carefully - I'm trying out a new voice costume! 🎉",
            "My voice is playing dress-up right now! 👀",
            "I'm shape-shifting my voice like a voice wizard! ✨"
        ]
    
    def detect_emotion(self, text):
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
        return "calm"

    def synthesize(self, text, emotion=None):
        # Detect emotion if not provided
        if emotion is None:
            emotion = self.detect_emotion(text)
        
        # Get emotion parameters
        params = self.emotions.get(emotion, self.emotions["calm"])
        
        try:
            # Generate audio with Bark
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                audio_array = generate_audio(
                    text, 
                    history_prompt=params["voice_preset"]
                )
                sf.write(temp_file.name, audio_array, SAMPLE_RATE)
                return temp_file.name
        
        except Exception as bark_error:
            print(f"Bark TTS generation error: {bark_error}")
        
        try:
            # Fallback to Glow-TTS
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                tts.tts_to_file(
                    text=f"{random.choice(self.fallback_explanations)} {text}",
                    file_path=temp_file.name,
                    speed=params["speed"]
                )
                return temp_file.name
        
        except Exception as glow_error:
            print(f"Glow-TTS generation error: {glow_error}")
            
            try:
                # Final fallback to gTTS
                temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                tts_fallback = gTTS(
                    text=f"{random.choice(self.fallback_explanations)} {text}", 
                    lang='en', 
                    tld='com', 
                    slow=False
                )
                tts_fallback.save(temp_file.name)
                return temp_file.name
            
            except Exception as final_error:
                print(f"Final TTS fallback error: {final_error}")
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
            # Use OpenAI Whisper for transcription
            result = whisper_model.transcribe(audio_file)
            text = result["text"].strip()
            return text
        except Exception as e:
            print(f"Whisper transcription error: {e}")
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
    # Capture launch output so we can print the URLs
    launch_info = demo.launch(share=True)
    # Depending on your Gradio version, launch_info might be a tuple or dict. Try printing it:
    print("Gradio launch info:", launch_info)

if __name__ == "__main__":
    main()
