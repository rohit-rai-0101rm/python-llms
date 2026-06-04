from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

pipe = pipeline("image-text-to-text", model="google/gemma-3-4b-it")

# --- Text-only chat ---
text_messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is the capital of France?"}
        ]
    }
]

result = pipe(text=text_messages, max_new_tokens=200)
print("Text response:", result[0]["generated_text"][-1]["content"])


