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


response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {
            "role": "system",
            "content": "You are an expert i maths and only ans maths related questions and if not related mathd say sorry"
        },
        {
            "role": "user",
            "content": "can u help em solve the a +b whole square"
        }
    ]
)

print(response.choices[0].message.content)
