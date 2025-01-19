import gradio as gr
import google.generativeai as genai
from elevenlabs import Voice, VoiceSettings, generate as elevenlabs_generate, set_api_key
import os
from pathlib import Path
import json
import random
from datetime import datetime

# Configuration
GEMINI_API_KEY = "AIzaSyC1y2piZc_Km6rBQ2iIhMIcHpyaI0hCQ3k"
ELEVENLABS_API_KEY = "sk_54aa1d19f315a8e2ba4a7175236497cb9b8e70bff61fd127"

# Initialize APIs
genai.configure(api_key=GEMINI_API_KEY)
set_api_key(ELEVENLABS_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Voice Configuration
BUDDY_VOICE = Voice(
    voice_id="bella",
    settings=VoiceSettings(stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True)
)

# [Previous character configuration and guidelines remain the same]
BUDDY_NAME = "Buddy"
MAX_RESPONSE_LENGTH = 200

SAFETY_GUIDELINES = """
Safety and Content Rules:
1. No scary, violent, or upsetting content
2. Never share personal information
3. No complex or technical language
4. Avoid mentions of money or purchases
5. No references to social media or online platforms
6. Keep all content age-appropriate (ages 4-10)
7. No medical or health advice
8. Redirect sensitive questions to parents/teachers
"""

EDUCATIONAL_THEMES = {
    "nature": ["Forest friends", "Garden discoveries", "Weather wonders", "Ocean adventures"],
    "science": ["Rainbow magic", "Plant growing", "Stars and moon", "Simple experiments"],
    "creative": ["Drawing and colors", "Music and sounds", "Imagination games", "Story time"],
    "life_skills": ["Being kind", "Helping at home", "Making friends", "Taking care of nature"]
}

PERSONALITY_RESPONSES = {
    "greetings": [
        "*happy bear hug* Hi friend!",
        "*wiggles ears* Hello there!",
        "*gentle smile* Welcome back!",
        "*playful wave* Hi buddy!"
    ],
    "closings": [
        "*happy bear dance* Bye for now!",
        "*soft bear wave* See you soon!",
        "*gentle smile* Take care, friend!",
        "*warm hug* Until next time!"
    ],
    "encouragements": [
        "What do you think about that?",
        "Would you like to try it too?",
        "Isn't learning fun?",
        "What's your favorite part?"
    ]
}

class BuddyBear:
    def __init__(self):
        self.chat_history = {}
        self.current_theme = None
        self.used_themes = set()
        
    def get_random_response(self, response_type):
        return random.choice(PERSONALITY_RESPONSES[response_type])
    
    def get_new_theme(self):
        available_themes = [
            theme for category in EDUCATIONAL_THEMES.values()
            for theme in category
            if theme not in self.used_themes
        ]
        if not available_themes or len(self.used_themes) >= 15:
            self.used_themes.clear()
            available_themes = [theme for category in EDUCATIONAL_THEMES.values() for theme in category]
        
        theme = random.choice(available_themes)
        self.used_themes.add(theme)
        return theme

    def format_response(self, user_input, username):
        context = f"""
        You are {BUDDY_NAME}, a friendly bear chatting with a child named {username}.
        Current theme: {self.get_new_theme()}
        
        Follow these rules:
        {SAFETY_GUIDELINES}
        
        Create a response that:
        1. Starts with: {self.get_random_response('greetings')}
        2. Includes a short, fun message about the theme
        3. Ends with: {self.get_random_response('encouragements')}
        4. Closes with: {self.get_random_response('closings')}
        
        Keep total response under {MAX_RESPONSE_LENGTH} characters.
        
        Child's message: {user_input}
        """
        
        response = model.generate_content(context)
        return response.text

    def generate_voice(self, text, username):
        try:
            # Generate audio using updated ElevenLabs API
            audio = elevenlabs_generate(
                text=text,
                voice=BUDDY_VOICE,
                model="eleven_multilingual_v1"
            )
            
            # Save the audio file
            audio_path = f"responses/{username}_latest.mp3"
            with open(audio_path, 'wb') as f:
                f.write(audio)
            
            return audio_path
        except Exception as e:
            print(f"Voice generation error: {e}")
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
        response = self.buddy.format_response(message, username)
        audio_path = self.buddy.generate_voice(response, username)
        
        history.append(("user", message))
        history.append(("assistant", response))
        self.save_chat_history(history, username)
        
        return response, audio_path, history

    def create_interface(self):
        buddy_ascii = """
        ʕ •ᴥ• ʔ
        Hi! I'm Buddy
        the Learning Bear!
        """
        
        with gr.Blocks(theme=gr.themes.Soft()) as interface:
            gr.Markdown(f"```\n{buddy_ascii}\n```")
            
            with gr.Row():
                username = gr.Textbox(
                    label="What's your name?",
                    placeholder="Enter your name",
                    scale=2
                )
            
            chatbot = gr.Chatbot(
                label="Chat with Buddy!",
                bubble_full_width=False,
                height=400
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Type your message here",
                    placeholder="What would you like to learn about?",
                    scale=3
                )
                clear = gr.Button("Clear Chat")
            
            audio_output = gr.Audio(label="Buddy's Voice")
            
            msg.submit(
                self.process_message,
                inputs=[msg, chatbot, username],
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