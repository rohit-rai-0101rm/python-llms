import warnings
import os
import json
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
2. If the question is not related to mathematics, reply politely.
3. ALWAYS return output in valid JSON format only.
4. Do not return any text outside JSON.

JSON Format:
{
  "solution": "string",
  "isMathsQuestion": true
}

Examples:

User: What is 5 + 7?
Assistant:
{
  "solution": "5 + 7 = 12",
  "isMathsQuestion": true
}

User: Solve x + 2 = 5
Assistant:
{
  "solution": "x = 3",
  "isMathsQuestion": true
}

User: Tell me a joke
Assistant:
{
  "solution": "Sorry, I can only answer mathematics-related questions.",
  "isMathsQuestion": false
}
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
            "content": "what is trigonometry"
        }
    ]
)

result = response.choices[0].message.content


parsed = json.loads(result)

print(parsed)