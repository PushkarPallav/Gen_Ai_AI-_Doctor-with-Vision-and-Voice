import os
import base64
from dotenv import load_dotenv
from groq import Groq

# Step 1: Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found. Please set it in your .env file.")

# Step 2: Encode image to base64
def encode_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image file not found: {image_path}")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Step 3: Analyze image with Groq multimodal model
def analyze_image_with_query(query, model, image_path):
    encoded_image = encode_image(image_path)
    client = Groq(api_key=GROQ_API_KEY)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            messages=messages,
            model=model
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error during Groq API call: {e}"

# Step 4: Run the analysis
if __name__ == "__main__":
    query = "Is there something wrong with my face?"
    image_path = "acne.jpg"
    model = "meta-llama/llama-4-scout-17b-16e-instruct"

    result = analyze_image_with_query(query, model, image_path)
    print(result)