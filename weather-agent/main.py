import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
load_dotenv()

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
def get_weather(city:str):
    url=f"https://wttr.in/{city.lower()}?format=%C+%t"
    response =requests.get(url)

    if response.status_code==200:
        return f"the weather in {city} is{response.text}"
    return "something went wrong"
    
    
    
def main():
    user_query = input("> ")

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
          
            {
                "role": "user",
                "content": user_query
            }
        ]
    )

    print(response.choices[0].message.content)
print(get_weather("uttar pradesh"))
