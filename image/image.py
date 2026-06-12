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

IMAGE_URL = "https://www.shutterstock.com/image-photo/indian-man-selling-vegetables-potato-260nw-2399360241.jpg" 

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
