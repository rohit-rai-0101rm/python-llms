import warnings
import os
from dotenv import load_dotenv
from openai import OpenAI

warnings.filterwarnings("ignore")

# Load API key from .env
load_dotenv()

# Use OpenAI SDK with Gemini's OpenAI-compatible endpoint
client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Send chat completion request to Gemini via OpenAI-compatible API
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Explain to me how AI works"
        }
    ]
)

print(response.choices[0].message.content)
