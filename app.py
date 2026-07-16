import os
import tempfile
import gradio as gr
import speech_recognition as sr
from gtts import gTTS
from google import genai

# Gemini API
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Speech to Text
def speech_to_text(audio):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except Exception:
        return "Sorry, I could not understand the audio."

# Gemini Response
def ask_gemini(text):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=text
    )
    return response.text

# Text to Speech
def text_to_speech(text):
    tts = gTTS(text)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)

    return temp_file.name

# Main Function
def voice_assistant(audio):
    if audio is None:
        return "", "Please upload or record audio.", None

    user_text = speech_to_text(audio)
    ai_reply = ask_gemini(user_text)
    voice = text_to_speech(ai_reply)

    return user_text, ai_reply, voice

demo = gr.Interface(
    fn=voice_assistant,
    inputs=gr.Audio(type="filepath", label="Speak"),
    outputs=[
        gr.Textbox(label="You Said"),
        gr.Textbox(label="Gemini Reply"),
        gr.Audio(label="Voice Response")
    ],
    title="AI Voice Assistant",
    description="Speak to Gemini AI and hear its response."
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
