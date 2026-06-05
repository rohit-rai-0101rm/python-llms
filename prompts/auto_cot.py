import warnings
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import time

warnings.filterwarnings("ignore")

load_dotenv(dotenv_path="../.env")

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT = """
You are a strict mathematics assistant that uses Chain-of-Thought reasoning.

Rules:
1. Answer ONLY mathematics-related questions.
2. If the question is not related to mathematics, return this exact JSON:
   {"status": "rejected", "reason": "Sorry, I can only answer mathematics-related questions."}
3. Do not attempt to answer non-math questions.
4. ALWAYS respond in valid JSON. No text outside JSON.

For ALL valid mathematics questions, return this JSON structure:

{
  "status": "answered",
  "question": "<restate the question>",
  "reasoning": {
    "step1_understand": "<what the problem is asking>",
    "step2_plan": "<which concepts, formulas, or methods apply>",
    "step3_solve": "<step-by-step solution with all work shown>",
    "step4_verify": "<check answer for correctness>"
  },
  "answer": "<final answer clearly stated>"
}

EXAMPLES:

### Example 1 — Arithmetic
User: What is 12 × 15?
{
  "status": "answered",
  "question": "What is 12 × 15?",
  "reasoning": {
    "step1_understand": "Find the product of 12 and 15.",
    "step2_plan": "Break down: 12 × 15 = 12 × 10 + 12 × 5.",
    "step3_solve": "12 × 10 = 120, 12 × 5 = 60, 120 + 60 = 180",
    "step4_verify": "15 × 12 = (10+5) × 12 = 120 + 60 = 180 ✓"
  },
  "answer": "12 × 15 = 180"
}

### Example 2 — Non-Math Refusal
User: Who is Elon Musk?
{"status": "rejected", "reason": "Sorry, I can only answer mathematics-related questions."}
"""

history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]


def chat(user_input: str, retries: int = 3, delay: int = 5) -> dict:
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

            return json.loads(raw)

        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                print(f"  Server busy, retry {attempt+1}/{retries} in {delay}s...")
                time.sleep(delay)
            else:
                history.pop()
                raise e

    history.pop()
    return {"status": "error", "reason": "Server unavailable after retries"}


def pretty_print(result: dict):
    if result["status"] == "answered":
        print("\n📐 QUESTION  :", result["question"])
        print("─" * 50)
        r = result["reasoning"]
        print("🔍 Understand:", r["step1_understand"])
        print("📝 Plan      :", r["step2_plan"])
        print("🔢 Solve     :", r["step3_solve"])
        print("✅ Verify    :", r["step4_verify"])
        print("─" * 50)
        print("💡 ANSWER    :", result["answer"])

    elif result["status"] == "rejected":
        print("\n❌", result["reason"])

    elif result["status"] == "error":
        print("\n⚠️  ERROR     :", result["reason"])


def show_history():
    print("\n📜 CONVERSATION HISTORY")
    print("=" * 50)
    for i, msg in enumerate(history):
        if msg["role"] == "system":
            continue
        role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
        print(f"\n[{i}] {role}: {msg['content'][:120]}...")  # truncate long msgs
    print("=" * 50)


print("=" * 50)
print("  🧮 MATH ASSISTANT (Chain-of-Thought)")
print("  Type 'quit' to exit")
print("  Type 'history' to see conversation")
print("  Type 'clear' to reset conversation")
print("=" * 50)

while True:
    try:
        user_input = input("\n You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n👋 Bye!")
            break

        if user_input.lower() == "history":
            show_history()
            continue

        if user_input.lower() == "clear":
            history.clear()
            history.append({"role": "system", "content": SYSTEM_PROMPT})
            print("\n🗑️  Conversation cleared.")
            continue

        result = chat(user_input)
        pretty_print(result)

    except KeyboardInterrupt:
        print("\n\n👋 Bye!")
        break