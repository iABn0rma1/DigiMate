import os
import json
import requests
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import google.generativeai as genai

# ----------------- Configure API -----------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

FLASH_API = os.getenv("GOOGLE_API_KEY")
FINE_API = os.getenv("FINESHARE_API_KEY")
if not FLASH_API or not FINE_API:
    raise EnvironmentError("Required API keys are not set in the environment variables.")

genai.configure(api_key=FLASH_API)
model = genai.GenerativeModel("gemini-1.5-flash")

# ----------------- Constants -----------------
PET_NAME = "Snowbell"

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

# ----------------- Helper Function -----------------
def generate_audio(text: str) -> str:
    """
    Sends the generated text to the FineShare API for text-to-speech conversion.
    Returns the audio file URL if successful.
    """
    url = "https://ttsapi.fineshare.com/v1/text-to-speech"
    headers = {
        "accept": "text/plain",
        "x-api-key": FINE_API,
        "Content-Type": "application/json"
    }
    payload = {
        "voice": "whiskers-digitalpet-21143",
        "amotion": "cheerful",
        "format": "mp3",
        "speech": text
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        response_data = json.loads(response.text)
        return response_data.get("downloadUrl", "")
    else:
        raise HTTPException(status_code=500, detail=f"TTS API Error: {response.text}")

# ----------------- FastAPI Endpoints -----------------
class PromptRequest(BaseModel):
    topic: str = "general"

@app.get("/", response_class=HTMLResponse)
def homepage(request: Request):
    endpoints = [
        {"name": "App Launch", "path": "/launch", "description": "Generates a welcome story when the app launches.",
         "method": "GET", "params": [], "syntax": "/launch"},
        
        {"name": "Generate Random", "path": "/generate-random", "description": "Generates a random puzzle or fun learning activity for kids.",
         "method": "GET", "params": [], "syntax": "/generate-random"},
        
        {"name": "Generate Story", "path": "/generate-story", "description": "Generates a story based on user-provided topics. (POST request)",
         "method": "POST", "params": ["topic (string)"], "syntax": """/generate-story{"topic":"chose_any_topic"}"""},

        {"name": "Query", "path": "/query", "description": "Answers specific user questions in a playful, engaging manner.",
         "method": "POST", "params": ["question (string)"], "syntax": """/query{"question":"ask_any_question"}"""},

        {"name": "Ask Kids", "path": "/ask-kids", "description": "Generates playful questions to engage kids.",
         "method": "GET", "params": [], "syntax": "/ask-kids"}
    ]
    return templates.TemplateResponse("homepage.html", {"request": request, "endpoints": endpoints})

@app.get("/launch")
def app_launch():
    try:
        response = model.generate_content(PROMPT_TEMPLATE)
        text = response.text.strip()
        audio_url = generate_audio(text)
        return {"message": "App launched successfully!", "result": text, "audio_url": audio_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/generate-random")
def generate_random():
    try:
        random_prompt = f"{PROMPT_TEMPLATE} Make a random puzzle or fun learning activity for kids!"
        response = model.generate_content(random_prompt)
        text = response.text.strip()
        audio_url = generate_audio(text)
        return {"result": text, "audio_url": audio_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/generate-story")
def generate_story(request: PromptRequest):
    try:
        user_prompt = f"{PROMPT_TEMPLATE} Today's topic is {request.topic}. Generate a story about it!"
        response = model.generate_content(user_prompt)
        text = response.text.strip()
        audio_url = generate_audio(text)
        return {"result": text, "audio_url": audio_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/query")
def query_user(question: str = Query(..., description="The specific query or question from the user")):
    try:
        query_prompt = f"""
        Act as {PET_NAME}, the friendly, adventurous cat tutor, and answer the following question in a playful, engaging, and age-appropriate manner for kids:
        '{question}'
        MUST FOLLOW:
        {GUIDELINES}
        """
        response = model.generate_content(query_prompt)
        text = response.text.strip()
        audio_url = generate_audio(text)
        return {"question": question, "result": text, "audio_url": audio_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/ask-kids")
def ask_kids():
    try:
        prompt = f"""
        Act as a playful, curious tutor for kids under 10 years old. 
        First, randomly choose an imaginative theme, such as space adventures, underwater worlds, magic forests, playful animals, or any other fun idea that sparks creativity.

        Then, generate a cheerful, age-appropriate question inspired by that theme that encourages kids to share their thoughts, imagination, or daily activities.

        MUST FOLLOW:
        {GUIDELINES}
        """
        response = model.generate_content(prompt)
        text = response.text.strip()
        audio_url = generate_audio(text)
        return {"result": text, "audio_url": audio_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
