import os
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Imports from your modular files
from analyze_image import encode_image, analyze_image_with_query
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_elevenlabs_rest

# System prompt for the doctor
system_prompt = """You have to act as a professional doctor, I know you are not but this is for learning purpose.
What's in this image? Do you find anything wrong with it medically?
If you make a differential, suggest some remedies for them. Do not add any numbers or special characters in
your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
Do not say 'In the image I see' but say 'With what I see, I think you have ....'
Don't respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot,
Keep your answer concise (max 2 sentences). No preamble, start your answer right away please."""

# Main processing function
def process_inputs(audio_filepath, image_filepath):
    # Step 1: Transcribe audio
    speech_to_text_output = transcribe_with_groq(
        stt_model="whisper-large-v3",
        audio_filepath=audio_filepath,
        api_key=GROQ_API_KEY
    )

    # Step 2: Analyze image
    if image_filepath:
        doctor_response = analyze_image_with_query(
            query=system_prompt + speech_to_text_output,
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            image_path=image_filepath
        )
    else:
        doctor_response = "No image provided for me to analyze."

    # Step 3: Convert doctor's response to speech
    output_audio_path = "final.mp3"
    text_to_speech_with_elevenlabs_rest(
        input_text=doctor_response,
        output_filepath=output_audio_path,
        voice_name="Aria",
        autoplay=False
    )

    return speech_to_text_output, doctor_response, output_audio_path

# Gradio interface (simple, no CSS)
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="🎙️ Record Your Concern"
        ),
        gr.Image(
            type="filepath",
            label="📷 Upload Image of Concern"
        )
    ],
    outputs=[
        gr.Textbox(
            label="🧠 Speech to Text",
            lines=4,         
            max_lines=8
        ),
        gr.Textbox(
            label="👨‍⚕️ Doctor's Response",
            lines=12,         
            max_lines=30      
        ),
        gr.Audio(
            label="🔊 Voice of the Doctor",
            type="filepath"
        )
    ],
    title="🩺 AI Doctor with Vision and Voice",
    description="Speak your symptoms and upload an image. The AI doctor will analyze and respond with voice."
)

iface.launch(debug=True)
