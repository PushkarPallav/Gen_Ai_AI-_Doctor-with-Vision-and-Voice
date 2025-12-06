import os
import logging
from io import BytesIO
from dotenv import load_dotenv
import speech_recognition as sr
from pydub import AudioSegment
from groq import Groq

# Step 0: Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found. Please set it in your .env file.")

# Step 1: Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 2: Record audio and save as MP3
def record_audio(file_path, timeout=20, phrase_time_limit=None):
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            logging.info("🎙️ Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("🗣️ Start speaking now...")
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("✅ Recording complete.")

            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            logging.info(f"📁 Audio saved to {file_path}")
    except Exception as e:
        logging.error(f"❌ Error during recording: {e}")

# Step 3: Transcribe audio using Groq Whisper
def transcribe_with_groq(stt_model, audio_filepath, api_key):
    try:
        client = Groq(api_key=api_key)
        with open(audio_filepath, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=stt_model,
                file=audio_file,
                language="en"
            )
        return transcription.text
    except Exception as e:
        logging.error(f"❌ Error during transcription: {e}")
        return None

# Step 4: Run the pipeline
if __name__ == "__main__":
    audio_filepath = "patient_voice_test_for_patient.mp3"
    stt_model = "whisper-large-v3"

    # Uncomment to record audio
    record_audio(file_path=audio_filepath)

    # Transcribe recorded audio
    result = transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY)
    if result:
        print("\n🧠 Transcription Result:")
        print(result)