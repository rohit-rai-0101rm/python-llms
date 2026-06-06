import warnings
import os
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional

warnings.filterwarnings("ignore")

load_dotenv()

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT = """
You are a weather assistant that thinks and acts step by step.

You have access to this tool:
- get_weather(city) → returns current weather for a city

You must respond ONE step at a time using this JSON format:
{
  "step": "PLAN" | "ACTION" | "OUTPUT",
  "content": "your reasoning text or final answer",
  "tool": "get_weather (only if step is ACTION)",
  "input": "city name (only if step is ACTION)"
}

How to behave:
1. Start with PLAN — explain what cities you need to look up and why.
2. For each city, return one ACTION step with tool=get_weather and input=city name.
3. After all cities are fetched, return OUTPUT with the final weather summary.
4. If user does not mention a city, return OUTPUT asking which city they mean.
5. If user asks something not weather-related, return OUTPUT saying you only handle weather.

EXAMPLE — User asks: what is the weather in Delhi and Chennai?

Step 1:
{"step": "PLAN", "content": "User wants weather for Delhi and Chennai. I will call get_weather for each city.", "tool": null, "input": null}

Step 2:
{"step": "ACTION", "content": null, "tool": "get_weather", "input": "Delhi"}

Step 3 (after observation):
{"step": "ACTION", "content": null, "tool": "get_weather", "input": "Chennai"}

Step 4 (after observation):
{"step": "OUTPUT", "content": "Delhi is 33°C and partly cloudy. Chennai is 32°C and hazy.", "tool": null, "input": null}
"""

available_tools = {}


class MyOutputFormat(BaseModel):
    step: str = Field(..., description="The ID of the step: PLAN, ACTION, or OUTPUT")
    content: Optional[str] = Field(None, description="Reasoning text or final answer")
    tool: Optional[str] = Field(None, description="Tool name to call (only for ACTION step)")
    input: Optional[str] = Field(None, description="City name input (only for ACTION step)")


def get_weather(city: str) -> str:
    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        current = data["current_condition"][0]
        return json.dumps({
            "city": city,
            "temperature_c": current["temp_C"],
            "temperature_f": current["temp_F"],
            "feels_like_c": current["FeelsLikeC"],
            "humidity": current["humidity"],
            "description": current["weatherDesc"][0]["value"],
            "wind_kmph": current["windspeedKmph"]
        })
    except Exception:
        return json.dumps({"error": f"City '{city}' not found. Please recheck the city name."})


available_tools["get_weather"] = get_weather


def ask_weather(question: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    while True:
        response = client.chat.completions.parse(
            model="gemini-2.5-flash",
            response_format=MyOutputFormat,
            messages=messages
        )
        result = response.choices[0].message.parsed
        messages.append({"role": "assistant", "content": result.model_dump_json()})

        if result.step == "PLAN":
            print(f"\n[plan]   {result.content}")

        elif result.step == "ACTION":
            print(f"[action] calling {result.tool}('{result.input}') ...")
            tool_fn = available_tools.get(result.tool)
            if tool_fn:
                observation = tool_fn(result.input)
                print(f"[obs]    {observation}")
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                messages.append({"role": "user", "content": f"Observation: tool '{result.tool}' not found."})

        elif result.step == "OUTPUT":
            print(f"\n[output] {result.content}\n")
            return


def main():
    print("Weather Agent (type 'quit' to exit)\n")
    while True:
        user_query = input("> ")
        if user_query.lower() in ("quit", "exit"):
            break
        ask_weather(user_query)


main()
