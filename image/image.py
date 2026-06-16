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

IMAGE_URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQL8vSEHMcX9kr-_MY0q7p294hU8XgqTOXJ7w&s" 

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": IMAGE_URL
                    }
                },
                {
                    "type": "text",
                    "text": "What is in this image? Describe it in detail."
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
