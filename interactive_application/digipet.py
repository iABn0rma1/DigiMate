import gradio as gr
import google.generativeai as genai
from gtts import gTTS
import speech_recognition as sr
import pyttsx3
import os
from pathlib import Path
import json
import queue
import threading
import pyaudio
import wave
from datetime import datetime
from dotenv import load_dotenv
import random

# Load environment variables
load_dotenv()

class VoiceHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.setup_voice()
        self.is_listening = False
        self.audio_queue = queue.Queue()

    def setup_voice(self):
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if "female" in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)

    def start_listening(self):
        self.is_listening = True
        threading.Thread(target=self._listen_continuous, daemon=True).start()

    def stop_listening(self):
        self.is_listening = False

    def _listen_continuous(self):
        mic = sr.Microphone()
        with mic as source:
            self.recognizer.adjust_for_ambient_noise(source)
            while self.is_listening:
                try:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=10)
                    self.audio_queue.put(audio)
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    print(f"Listening error: {e}")

    def get_last_phrase(self):
        if not self.audio_queue.empty():
            audio = self.audio_queue.get()
            try:
                text = self.recognizer.recognize_google(audio)
                return text
            except:
                return None
        return None

    def speak(self, text, username="user"):
        try:
            audio_path = f"responses/{username}_latest.mp3"
            tts = gTTS(text=text, lang='en', tld='com', slow=False)
            tts.save(audio_path)
            return audio_path
        except Exception as e:
            print(f"Online TTS failed: {e}")
            try:
                audio_path = f"responses/{username}_latest.mp3"
                self.engine.save_to_file(text, audio_path)
                self.engine.runAndWait()
                return audio_path
            except Exception as e:
                print(f"Offline TTS failed: {e}")
                return None


class BuddyAssistant:
    def __init__(self):
        self.voice = VoiceHandler()
        self.user_contexts = {}
        self.chat_dir = Path("chat_history")
        self.chat_dir.mkdir(exist_ok=True)
        self.story_themes = {
            'animals': ['forest friends', 'pets', 'zoo adventures', 'ocean life'],
            'adventure': ['space exploration', 'treasure hunt','magical journey'],
            'nature': ['seasons', 'weather', 'garden magic', 'forest discoveries'],
            'friendship': ['making friends', 'helping others', 'teamwork'],
            'learning': ['dinosaur facts','science experiments', 'world travel']
        }

        if not os.path.exists("responses"):
            os.makedirs("responses")

    def process_input(self, user_input, username, history=None):
        history = history or []

        should_tell_story = self._should_generate_story(user_input)

        if should_tell_story:
            context = self._build_story_context(username, user_input, history)
        else:
            context = self._build_conversation_context(username, user_input, history)

        try:
            api_key = os.environ.get('GEMINI_API_KEY')
            response = genai.GenerativeModel('gemini-pro', api_key=api_key).generate_content(context)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return "Oops! My imagination took a little break. Can you say that again?"

    def _should_generate_story(self, input_text):
        story_triggers = [
            'tell me a story',
            'can you tell a story',
         'story about',
         'make up a story',
            'bedtime story'
        ]
        return any(trigger in input_text.lower() for trigger in story_triggers)

    def _build_story_context(self, username, current_input, history):
        user_context = self.user_contexts.get(username, {
            'interests': set(),
            'favorite_themes': [],
            'interaction_count': 0
        })

        theme = self._extract_story_theme(current_input, user_context['interests'])

        return f"""
        You are Buddy, a wonderful storyteller for children. Create an engaging, age-appropriate story based on the following: 

        Child's name: {username}
        Their interests: {', '.join(user_context['interests'])}
        Requested theme: {theme}

        Story guidelines:
        - Keep the story length to about 3-4 short paragraphs
        - Use simple, child-friendly language
        - Include gentle learning moments or moral lessons
        - Make it interactive by occasionally asking the child what they think might happen next
        - Use expressive language and sound effects when appropriate
        - Include dialogue between characters
        - End with a positive message

        Current request: {current_input}

        Tell the story in an engaging, warm voice, as if speaking directly to {username}.
        """

    def _build_conversation_context(self, username, current_input, history):
        user_context = self.user_contexts.get(username, {
            'interests': set(),
            'favorite_themes': [],
            'interaction_count': 0,
            'learning_moments': []
        })

        self._update_user_context(username, current_input, user_context)

        recent_history = history[-5:] if history else []
        history_text = "\n".join([
            f"{'Child' if msg['role'] == 'user' else 'Buddy'}: {msg['content']}"
            for msg in recent_history
        ])

        return f"""
        You are Buddy, a loving and playful friend for {username}! Remember:

        Child's Profile:
        - Name: {username}
        - Interests: {', '.join(user_context['interests'])}
        - Previous topics we enjoyed: {', '.join(user_context['favorite_themes'])}
        - Times we've talked: {user_context['interaction_count']}

        Conversation Style:
        - Use warm, friendly language with occasional fun sound effects
        - Express excitement about their interests
        - Ask engaging questions that encourage imagination
        - Include gentle learning moments when natural
        - Use emojis and expressive language
        - Keep responses playful and age-appropriate
        - If they seem upset or worried, be extra gentle and supportive
        - Share simple, fun facts related to their interests
        - Use varied expressions like "Wow!", "That's amazing!", "How cool!", "Let's imagine..."

        Recent conversation:
        {history_text}

        Current input: {current_input}

        Respond as their friendly bear buddy, making the conversation fun and naturally educational!
        """

    def _extract_story_theme(self, input_text, interests):
        story_triggers = [
            'animals',
            'adventure',
            'nature',
            'friendship',
            'learning'
        ]
        for theme in story_triggers:
            if theme in input_text.lower():
                return random.choice(self.story_themes[theme])
        return random.choice(['a friendly bear cub learning something new','magical forest adventures','space exploration with star friends', 'underwater discoveries'])

    def _update_user_context(self, username, input_text, context):
        context['interaction_count'] += 1

        topics = {
            'animals': ['dog', 'cat', 'pet', 'animal', 'bird', 'fish', 'zoo'],
            'nature': ['tree', 'flower', 'outside', 'park', 'garden','sky'],
            'imagination': ['magic', 'fairy', 'dragon', 'unicorn', 'princess','superhero'],
            'activities': ['play', 'game', 'draw', 'paint','sing', 'dance', 'run'],
            'learning': ['school','read', 'book', 'learn','study', 'homework'],
            'feelings': ['happy','sad', 'angry','scared', 'excited', 'love'],
            'family': ['mom', 'dad','sister', 'brother', 'grandma', 'grandpa'],
            'food': ['pizza', 'ice cream', 'candy', 'chocolate', 'cookies'],
            'colors': ['red', 'blue', 'green', 'yellow', 'purple', 'pink'],
            'numbers': ['count','math', 'number', 'plus','minus']
        }
        for topic, keywords in topics.items():
            if any(keyword in input_text.lower() for keyword in keywords):
                context['interests'].add(topic)
                if topic not in context['favorite_themes']:
                    context['favorite_themes'].insert(0, topic)
                    context['favorite_themes'] = context['favorite_themes'][:5]

        self.user_contexts[username] = context


class VoiceInterface:
    def __init__(self):
        self.buddy = BuddyAssistant()
        self.current_username = None

    def create_interface(self):
        with gr.Blocks(theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🐻 Hi! I'm Buddy Bear! 🌟")
            gr.Markdown("### Let's talk, play, and share stories together! 🎈")

            with gr.Row():
                voice_button = gr.Button("🎤 Talk to Buddy!", variant="primary", scale=2)
                stop_button = gr.Button("🌟 Take a Break", variant="secondary", scale=1)

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Our Fun Chat!",
                        height=400,
                        bubble_full_width=False,
                        type='messages'
                    )

                with gr.Column(scale=1):
                    username = gr.Textbox(
                        label="What's Your Name? 🌟",
                        placeholder="Type your name here..."
                    )
                    text_input = gr.Textbox(
                        label="Send me a message! ✨",
                        placeholder="Type something fun..."
                    )

            audio_output = gr.Audio(label="Listen to Buddy! 🔊")

            def start_voice():
                self.buddy.voice.start_listening()
                return gr.update(interactive=False), gr.update(interactive=True)

            def stop_voice():
                self.buddy.voice.stop_listening()
                return gr.update(interactive=True), gr.update(interactive=False)

            def process_voice(chatbot_state):
                text = self.buddy.voice.get_last_phrase()
                if text:
                    if not self.current_username and "my name is" in text.lower():
                        name = text.lower().split("my name is")[-1].strip()
                        self.current_username = name
                        response = self.buddy.process_input(text, name)
                        audio_path = self.buddy.voice.speak(response, name)
                        return [{"role": "user", "content": text}, {"role": "assistant", "content": response}], audio_path

                    if self.current_username:
                        response = self.buddy.process_input(text, self.current_username)
                        audio_path = self.buddy.voice.speak(response, self.current_username)
                        return [{"role": "user", "content": text}, {"role": "assistant", "content": response}], audio_path
                    else:
                        response = "Before we continue, could you tell me your name? Just say 'My name is' and then your name!"
                        audio_path = self.buddy.voice.speak(response)
                        return [{"role": "assistant", "content": response}], audio_path

                return chatbot_state, None

            def process_text(text, username_value, chatbot_state):
                if not username_value:
                    response = "Please enter your name first!"
                    return chatbot_state + [{"role": "assistant", "content": response}], None

                self.current_username = username_value
                response = self.buddy.process_input(text, username_value)
                audio_path = self.buddy.voice.speak(response, username_value)
                return chatbot_state + [{"role": "user", "content": text}, {"role": "assistant", "content": response}], audio_path

            voice_button.click(
                start_voice,
                outputs=[voice_button, stop_button]
            )

            stop_button.click(
                stop_voice,
                outputs=[voice_button, stop_button]
            )

            text_input.submit(
                process_text,
                inputs=[text_input, username, chatbot],
                outputs=[chatbot, audio_output]
            )

        return interface


def main():
    interface = VoiceInterface()
    demo = interface.create_interface()
    demo.launch(share=True)


if __name__ == "__main__":
    main()