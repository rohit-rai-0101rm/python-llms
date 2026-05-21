import warnings
import os
from dotenv import load_dotenv
from openai import OpenAI

warnings.filterwarnings("ignore")


load_dotenv()


client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
SYSTEM_PROMPT = """
You are a strict mathematics assistant.

Rules:
1. Answer ONLY mathematics-related questions.
2. If the question is not related to mathematics, reply exactly with:
"Sorry, I can only answer mathematics-related questions."
3. Do not explain why.
4. Do not attempt to answer non-math questions.
"""
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {
            "role": "system",
            "content":SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": "tell me a joke"
        }
    ]
)

print(response.choices[0].message.content)
