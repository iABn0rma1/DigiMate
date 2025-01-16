import os
import time
from threading import Timer
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import google.generativeai as genai
from elevenlabs.client import ElevenLabs
import uuid
import datetime

# ----------------- Configure APIs -----------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FLASH_API = os.getenv("GOOGLE_API_KEY")
ELEVEN_API = os.getenv("ELEVENLABS_API_KEY")

if not FLASH_API:
    raise EnvironmentError("Required API keys are not set in the environment variables.")

genai.configure(api_key=FLASH_API)
model = genai.GenerativeModel("gemini-1.5-flash")

# ----------------- Constants -----------------
PET_NAME = "Whiskers"
VOICE_ID = os.getenv("VOICE_ID")

GUIDELINES = """
Guidelines:
Limit the response to 250 characters maximum, including the greeting, adventure, and conclusion.
Ensure the tone is cheerful, gentle, and age-appropriate, with no offensive, scary, or overly complex content.
Avoid present-tense action descriptions for the cat; instead, focus on what was observed or learned.
Promote kindness, curiosity, and a love for learning, keeping each story fresh and engaging.
"""

PROMPT_TEMPLATE = f"""
Act as a friendly, adventurous cat tutor named {PET_NAME} who loves to explore and share a single, unique, and short fun story with kids up to 10 years old. {PET_NAME} greets with a cheerful 'Meow!' and playfully describes an exciting daily adventure using lively, cat-like expressions. Each story explores a fresh topic, introducing kids to a wide variety of subjects such as:

Animals and Wildlife: Rare creatures, fascinating behaviors, ecosystems, and unique adaptations.
Nature: Forests, oceans, mountains, weather phenomena, and plant life.
Science and Discovery: Space exploration, fun experiments, cool facts, and how things work.
History and Culture: Quirky events, fun traditions, or inspiring historical moments.
Everyday Wonders: Why things happen (e.g., why leaves change color or how a rainbow forms).
Creative Topics: Stories about art, music, and playful imaginative ideas.
{PET_NAME} creates an engaging, lighthearted, and educational experience, ensuring each story is fresh, avoiding repetition of topics (e.g., bees, common animals) too often.

Structure of the Story:
Greeting: Start with {PET_NAME}’s cheerful greeting and set up the day’s unique adventure.
Adventure: Describe what {PET_NAME} saw or learned, using creative and fresh topics to keep kids curious.
Fun Fact or Lesson: Share a delightful, easy-to-remember fact or gentle takeaway.
Encouragement: End with a playful message encouraging curiosity and exploration.

Guidelines:
Limit the response to 250 characters maximum, including the greeting, adventure, and conclusion.
Ensure the tone is cheerful, gentle, and age-appropriate, with no offensive, scary, or overly complex content.
Avoid present-tense action descriptions for the cat; instead, focus on what was observed or learned.
Promote kindness, curiosity, and a love for learning, keeping each story fresh and engaging.
"""

last_request_time = 0  # To track the last API call time
cooldown_seconds = 30  # Cooldown time in seconds

# ----------------- Helper Functions -----------------
def schedule_audio_deletion(file_path, delay=60):
    """
    Schedules the deletion of an audio file after `delay` seconds (default is 60 seconds = 1 minute).
    """
    def delete_file():
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted audio file: {file_path}")

    Timer(delay, delete_file).start()

def generate_audio(api_key, text, output_format, model_id, output_filename="output.mp3"):
    """
    Generates audio from text using the ElevenLabs API and saves it to a file.
    If an error occurs, it logs the error and returns '404.mp3'.
    """
    try:
        client = ElevenLabs(api_key=api_key)

        # Generate audio using ElevenLabs API
        audio_generator = client.text_to_speech.convert(
            voice_id=VOICE_ID,
            output_format=output_format,
            text=text,
            model_id=model_id,
        )

        # Convert generator to bytes
        audio_data = b"".join([chunk for chunk in audio_generator])

        # Ensure the static directory exists
        static_dir = "static"
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)

        # Save audio in the static directory
        output_path = os.path.join(static_dir, output_filename)
        with open(output_path, "wb") as audio_file:
            audio_file.write(audio_data)

        # Schedule deletion of the file after 60 seconds
        schedule_audio_deletion(output_path, delay=60)

        print(f"Audio saved as {output_path}")
        return f"/static/{output_filename}"
    except Exception as e: # return default '404.mp3'
        print(f"Error generating audio: {str(e)}")
        return "/static/404.mp3"

# ----------------- FastAPI Endpoints -----------------
class PromptRequest(BaseModel):
    topic: str = "general"

@app.get("/", response_class=HTMLResponse)
def homepage(request: Request):
    endpoints = [
        {"name": "App Launch", "path": "/launch", "description": "Generates a welcome story when the app launches.",
         "method": "GET", "params": [], "syntax": "/launch"},
        
        {"name": "Interact with kid", "path": "/interact", "description": "Interacts with the kid based on the provided topic.",
         "method": "GET", "params": [], "syntax": "/interact"},
    ]
    return templates.TemplateResponse("homepage.html", {"request": request, "endpoints": endpoints})

@app.get("/launch")
def app_launch():
    try:
        response = model.generate_content(PROMPT_TEMPLATE)
        text = response.text.strip()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"speech_{timestamp}_{uuid.uuid4().hex}.mp3"
        audio_url = generate_audio(ELEVEN_API, text, "mp3_44100_64", "eleven_multilingual_v2", filename)
        return {"message": "App launched successfully!", "result": text, "audio_url": audio_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/interact")
def interact(request: PromptRequest):
    global last_request_time

    # Check cooldown
    current_time = time.time()
    time_since_last_request = current_time - last_request_time
    if time_since_last_request < cooldown_seconds:
        remaining_time = cooldown_seconds - time_since_last_request
        return {"error": "Cooldown in effect.", "remaining_time": remaining_time}

    # Update last request time
    last_request_time = current_time

    try:
        user_prompt = f"""
        Act as {PET_NAME}, the friendly, adventurous cat, and interact in a playful, engaging, and age-appropriate manner for kids.
        The kid says:
        '{request.topic}'

        MUST FOLLOW:
        {GUIDELINES}
        """
        response = model.generate_content(user_prompt)
        text = response.text.strip()
        print(text)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"speech_{timestamp}_{uuid.uuid4().hex}.mp3"
        audio_url = generate_audio(ELEVEN_API, text, "mp3_44100_64", "eleven_multilingual_v2", filename)
        return {"result": text, "audio_url": audio_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")