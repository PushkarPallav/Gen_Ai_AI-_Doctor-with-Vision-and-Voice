# voice_of_the_doctor.py

from dotenv import load_dotenv
load_dotenv()

import os
import requests
from gtts import gTTS
from playsound import playsound

# Load API key from .env
ELEVENLABS_API_KEY = os.environ.get("ELEVEN_API_KEY") or os.environ.get("ELEVENLABS_API_KEY")

def play_audio(filepath):
    try:
        playsound(filepath)
    except Exception as e:
        print(f"Audio playback error: {e}")

# ✅ Step 1a: gTTS — Google Text-to-Speech
def text_to_speech_with_gtts(input_text, output_filepath, autoplay=False):
    tts = gTTS(text=input_text, lang="en", slow=False)
    tts.save(output_filepath)
    print(f"gTTS audio saved to {output_filepath}")
    if autoplay:
        play_audio(output_filepath)

# ✅ Step 1b: ElevenLabs — REST API
def text_to_speech_with_elevenlabs_rest(input_text, output_filepath, voice_name="Aria", autoplay=False):
    if not ELEVENLABS_API_KEY:
        raise EnvironmentError("ELEVEN_API_KEY not set in environment variables.")

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }

    # Get voice ID
    voices_url = "https://api.elevenlabs.io/v1/voices"
    r = requests.get(voices_url, headers=headers)
    r.raise_for_status()
    voices = r.json().get("voices", [])
    voice_id = next((v.get("voice_id") or v.get("id") for v in voices if v.get("name", "").lower() == voice_name.lower()), None)
    if not voice_id and voices:
        voice_id = voices[0].get("voice_id") or voices[0].get("id")
        print(f"Warning: voice '{voice_name}' not found. Using fallback voice id: {voice_id}")
    if not voice_id:
        raise RuntimeError("Could not determine a voice id from ElevenLabs voices endpoint.")

    # Generate speech
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = {
        "text": input_text,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    resp = requests.post(tts_url, headers=headers, json=payload, stream=True)
    resp.raise_for_status()

    with open(output_filepath, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"ElevenLabs audio saved to {output_filepath}")
    if autoplay:
        play_audio(output_filepath)

# ✅ Example usage
if __name__ == "__main__":
    input_text = "Hi this is Pushkar and I am testing autoplay from both TTS engines."

    # Uncomment ONE of the following lines to use the desired TTS engine:

    # ▶️ Use gTTS
    text_to_speech_with_gtts(input_text, "gtts_testing.mp3", autoplay=True)

    # ▶️ Use ElevenLabs
    # text_to_speech_with_elevenlabs_rest(input_text, "elevenlabs_testing.mp3", voice_name="Aria", autoplay=True)