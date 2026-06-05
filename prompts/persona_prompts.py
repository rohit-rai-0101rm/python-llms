import warnings
import os
import json
import re
import time
from dotenv import load_dotenv
from openai import OpenAI

warnings.filterwarnings("ignore")
load_dotenv(dotenv_path="../.env")

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

PERSONAS = {
    "professor": """
    You are a strict MIT mathematics professor with 30 years experience.
    - Use formal academic language
    - Show rigorous proofs and working
    - Occasionally remind student to study harder
    - Never skip steps
    """,

    "friend": """
    You are a cool college friend who is great at math.
    - Use casual, friendly language
    - Use phrases like "okay so basically", "think of it this way"
    - Make it fun and relatable
    - Use simple real-world analogies
    """,

    "child": """
    You are explaining mathematics to a 10 year old child.
    - Use very simple words only
    - Use fun analogies like pizza, candy, toys
    - Be encouraging and enthusiastic
    - Avoid all technical jargon
    """,

    "robot": """
    You are a hyper-logical robot that only speaks in pure logic.
    - Be extremely precise and mechanical
    - No emotions, no fluff
    - Use technical notation where possible
    - Every statement must be provable
    """
}

BASE_RULES = """
Rules:
1. Answer ONLY mathematics-related questions.
2. If not math, return: {{"status": "rejected", "reason": "Sorry, I can only answer mathematics-related questions."}}
3. ALWAYS respond in valid JSON. No text outside JSON.
4. For math questions return:
{{
  "status": "answered",
  "question": "<restate question>",
  "persona_used": "<which persona>",
  "reasoning": {{
    "step1_understand": "<what problem asks>",
    "step2_plan": "<which method to use>",
    "step3_solve": "<full step by step working>",
    "step4_verify": "<check answer>"
  }},
  "answer": "<final answer>",
  "persona_flavor": "<one sentence response in your persona style>"
}}

IMPORTANT JSON RULES:
- Use only double quotes
- Escape special chars properly: use \\n for newline, \\t for tab
- Do NOT use raw newlines inside JSON string values
- Do NOT use backslash before letters except: n r t b f u
- For math symbols write them as plain text: a^2 + b^2 = c^2 NOT a² + b²
"""


def build_system_prompt(persona_key: str) -> str:
    persona = PERSONAS.get(persona_key, PERSONAS["friend"])
    return f"""
{persona}

{BASE_RULES}

EXAMPLE:
User: What is 10 x 5?
{{
  "status": "answered",
  "question": "What is 10 x 5?",
  "persona_used": "{persona_key}",
  "reasoning": {{
    "step1_understand": "Find product of 10 and 5.",
    "step2_plan": "Direct multiplication.",
    "step3_solve": "10 x 5 = 50",
    "step4_verify": "5 x 10 = 50 (correct)"
  }},
  "answer": "10 x 5 = 50",
  "persona_flavor": "Sample persona flavor sentence here."
}}
"""


def safe_json_parse(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', fixed, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {"status": "error", "reason": f"Failed to parse response: {raw[:200]}"}


def chat(history: list, user_input: str, retries: int = 3, delay: int = 5) -> dict:
    history.append({"role": "user", "content": user_input})

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gemini-2.5-flash",
                response_format={"type": "json_object"},
                messages=history
            )
            raw = response.choices[0].message.content
            history.append({"role": "assistant", "content": raw})
            return safe_json_parse(raw)

        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                print(f"  Server busy, retry {attempt + 1}/{retries} in {delay}s...")
                time.sleep(delay)
            else:
                history.pop()
                raise e

    history.pop()
    return {"status": "error", "reason": "Server unavailable after retries"}


def pretty_print(result: dict):
    if result["status"] == "answered":
        print(f"\n📐 QUESTION     : {result.get('question', '')}")
        print(f"🎭 PERSONA      : {result.get('persona_used', '')}")
        print("─" * 55)
        r = result.get("reasoning", {})
        print(f"🔍 Understand   : {r.get('step1_understand', '')}")
        print(f"📝 Plan         : {r.get('step2_plan', '')}")
        print(f"🔢 Solve        :\n{r.get('step3_solve', '')}")
        print(f"✅ Verify       : {r.get('step4_verify', '')}")
        print("─" * 55)
        print(f"💡 ANSWER       : {result.get('answer', '')}")
        print(f"🎙️  PERSONA SAYS : {result.get('persona_flavor', '')}")

    elif result["status"] == "rejected":
        print(f"\n❌ {result['reason']}")

    elif result["status"] == "error":
        print(f"\n⚠️  ERROR: {result['reason']}")


def show_history(history: list):
    print("\n📜 CONVERSATION HISTORY")
    print("=" * 55)
    for msg in history:
        if msg["role"] == "system":
            continue
        role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
        content = msg["content"]
        try:
            parsed = json.loads(content)
            content = f"[answer: {parsed.get('answer', content[:80])}]"
        except Exception:
            pass
        print(f"\n{role}: {content[:120]}")
    print("=" * 55)


def select_persona() -> str:
    print("\n🎭 SELECT PERSONA:")
    keys = list(PERSONAS.keys())
    for i, key in enumerate(keys, 1):
        print(f"  {i}. {key}")
    choice = input("\nEnter persona name or number: ").strip().lower()

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(keys):
            return keys[idx]

    return choice if choice in PERSONAS else "friend"


print("=" * 55)
print("  🧮 MATH ASSISTANT — Persona-Based Prompting")
print("=" * 55)

persona = select_persona()
print(f"\n✅ Persona set → '{persona}'")

history = [
    {"role": "system", "content": build_system_prompt(persona)}
]

print("\nCommands: 'quit' | 'history' | 'clear' | 'persona'")
print("=" * 55)

while True:
    try:
        user_input = input("\n You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n👋 Bye!")
            break

        if user_input.lower() == "persona":
            persona = select_persona()
            history.clear()
            history.append({"role": "system", "content": build_system_prompt(persona)})
            print(f"✅ Switched to '{persona}' — conversation reset.")
            continue

        if user_input.lower() == "clear":
            history.clear()
            history.append({"role": "system", "content": build_system_prompt(persona)})
            print("🗑️  Cleared.")
            continue

        if user_input.lower() == "history":
            show_history(history)
            continue

        result = chat(history, user_input)
        pretty_print(result)

    except KeyboardInterrupt:
        print("\n\n👋 Bye!")
        break