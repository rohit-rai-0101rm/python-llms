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

Examples:

User: What is 5 + 7?
Assistant: 5 + 7 = 12

User: Solve x + 2 = 5
Assistant: x = 3

User: What is JavaScript?
Assistant: Sorry, I can only answer mathematics-related questions.

User: Tell me a joke
Assistant: Sorry, I can only answer mathematics-related questions.
"""

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": "what is trignometry"
        }
    ]
)

print(response.choices[0].message.content)