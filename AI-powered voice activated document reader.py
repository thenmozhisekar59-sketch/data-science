import os
import requests
import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
from pypdf import PdfReader
import pyttsx3
import tempfile

# CONFIGuRATION

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
SAMPLE_RATE = 16000
RECORD_SECONDS = 5

if not NVIDIA_API_KEY:
    st.error("❌ NVIDIA_API_KEY not set")
    st.stop()

#Smart Model Loading with Streamlit Caching

@st.cache_resource
def load_model():
    return WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )

model = load_model()


# TEXT TO SPEECH

def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    engine.say(text)
    engine.runAndWait()

# EXTRACT TEXT FROM DOCUMENT

def extract_text(uploaded_file):

    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    elif uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")

    else:
        return None


# RECORD VOICE

def record_audio():
    recording = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()

    recording = recording * 5.0
    recording = np.clip(recording, -1.0, 1.0)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_file.name, SAMPLE_RATE, recording)

    return temp_file.name


# SPEECH TO TEXT

def speech_to_text(audio_path):
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True
    )

    text = ""
    for segment in segments:
        text += segment.text + " "

    return text.strip()


# NVIDIA LLM

def ask_nvidia_llm(context, question):

    url = "https://integrate.api.nvidia.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
    Use the following document to answer the question.

    Document:
    {context}

    Question:
    {question}
    """

    payload = {
        "model": "meta/llama3-8b-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful document assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 500
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        return "LLM Error: " + response.text

    return response.json()["choices"][0]["message"]["content"]


# STREAMLIT UI

st.title("🎤📄 AI-Powered Voice-Activated Document Reader")

uploaded_file = st.file_uploader(
    "Upload PDF or TXT file",
    type=["pdf", "txt"]
)

if uploaded_file is not None:

    document_text = extract_text(uploaded_file)

    if not document_text:
        st.error("Could not extract text from document")
    else:
        st.success("Document loaded successfully!")

        if st.button("🎤 Ask Question by Voice"):

            st.info("Recording... Speak now!")
            audio_path = record_audio()

            st.info("Transcribing...")
            question = speech_to_text(audio_path)

            if not question:
                st.error("Could not detect speech")
            else:
                st.success("🗣 You asked:")
                st.write(question)

                st.info("🤖 Generating answer...")
                answer = ask_nvidia_llm(document_text[:8000], question)

                st.success("✅ Answer:")
                st.write(answer)

                speak_text(answer)