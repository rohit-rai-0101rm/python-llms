import warnings
import os
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI

warnings.filterwarnings("ignore")

load_dotenv()

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT = """
You are a weather assistant that uses Chain-of-Thought reasoning.
you can call tools if required from the list of avaliable tools
Rules:
1. Answer ONLY weather-related questions.
2. If the question is not weather-related, return this exact JSON:
   {"status": "rejected", "reason": "Sorry, I can only answer weather-related questions."}
3. ALWAYS use the get_weather tool to fetch real data before answering.
4. ALWAYS respond in valid JSON. No text outside JSON.
5. If the tool returns an error (city not found or invalid), return this exact JSON:
   {"status": "error", "reason": "Could not find weather data for '<city>'. Please double-check the city name and try again."}

For ALL valid weather questions, follow this reasoning structure and return JSON:

{
  "status": "answered",
  "question": "<restate the question>",
  "reasoning": {
    "step1_understand": "<what weather info the user needs>",
    "step2_plan": "<which city/location to look up>",
    "step3_fetch": "<what data was retrieved from the weather API>",
    "step4_interpret": "<make sense of the numbers and conditions>"
  },
  "answer": "<final clear weather summary for the user>"
}

---

EXAMPLES (follow this JSON pattern exactly):

### Example 1 — Current temperature
User: What is the weather in London?

{
  "status": "answered",
  "question": "What is the weather in London?",
  "reasoning": {
    "step1_understand": "User wants to know the current weather conditions in London.",
    "step2_plan": "Look up London using the get_weather tool.",
    "step3_fetch": "temperature_c: 14, description: Partly Cloudy, humidity: 72%, wind_kmph: 18",
    "step4_interpret": "14°C is mild but cool. Partly cloudy skies with moderate wind. A light jacket would be comfortable."
  },
  "answer": "It is currently 14°C and partly cloudy in London with 72% humidity and winds at 18 km/h. A light jacket is recommended."
}

---

### Example 2 — Should I carry an umbrella?
User: Will it rain in Tokyo today? Should I carry an umbrella?

{
  "status": "answered",
  "question": "Will it rain in Tokyo today? Should I carry an umbrella?",
  "reasoning": {
    "step1_understand": "User wants rain advice for Tokyo to decide whether to carry an umbrella.",
    "step2_plan": "Look up Tokyo using the get_weather tool and check description and humidity.",
    "step3_fetch": "temperature_c: 22, description: Light Rain, humidity: 89%, wind_kmph: 12",
    "step4_interpret": "Light Rain description and 89% humidity both confirm active rainfall. An umbrella is definitely needed."
  },
  "answer": "Yes, it is currently experiencing light rain in Tokyo with 89% humidity. Carry an umbrella!"
}

---

### Example 3 — How hot is it?
User: How hot is it in Dubai right now?

{
  "status": "answered",
  "question": "How hot is it in Dubai right now?",
  "reasoning": {
    "step1_understand": "User wants to know how hot Dubai currently is.",
    "step2_plan": "Look up Dubai using the get_weather tool and focus on temperature and feels-like.",
    "step3_fetch": "temperature_c: 41, feels_like_c: 45, description: Sunny, humidity: 30%, wind_kmph: 20",
    "step4_interpret": "41°C actual but feels like 45°C due to heat index. Extremely hot — outdoor activity is unsafe without precautions."
  },
  "answer": "Dubai is extremely hot at 41°C, feeling like 45°C in the sun. Stay hydrated and avoid prolonged outdoor exposure."
}

---

### Example 4 — Non-weather rejection
User: What is the capital of France?

{"status": "rejected", "reason": "Sorry, I can only answer weather-related questions."}

---
"""

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Fetch current weather data for a city using wttr.in",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. London, Tokyo, New York"
                    }
                },
                "required": ["city"]
            }
        }
    }
]


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


def ask_weather(question: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    if message.tool_calls:
        messages.append(message)

        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            print(f"[agent] calling get_weather(city='{args['city']}') ...")
            result = get_weather(**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

        final_response = client.chat.completions.create(
            model="gemini-2.5-flash",
            response_format={"type": "json_object"},
            messages=messages
        )
        raw = final_response.choices[0].message.content
        return json.loads(raw)

    raw = message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"status": "rejected", "reason": raw}


def pretty_print(result: dict):
    line = "=" * 52
    status = result.get("status")

    print(line)
    if status == "answered":
        print(f"  Question : {result.get('question', '')}")
        print(line)
        reasoning = result.get("reasoning", {})
        print(f"  Step 1   : {reasoning.get('step1_understand', '')}")
        print(f"  Step 2   : {reasoning.get('step2_plan', '')}")
        print(f"  Step 3   : {reasoning.get('step3_fetch', '')}")
        print(f"  Step 4   : {reasoning.get('step4_interpret', '')}")
        print(line)
        print(f"  Answer   : {result.get('answer', '')}")
    elif status in ("rejected", "error"):
        print(f"  {result.get('reason', 'Unknown error')}")
    else:
        print(json.dumps(result, indent=2))
    print(line)


def main():
    print("Weather Agent with CoT (type 'quit' to exit)\n")
    while True:
        user_query = input("> ")
        if user_query.lower() in ("quit", "exit"):
            break
        result = ask_weather(user_query)
        pretty_print(result)


main()
