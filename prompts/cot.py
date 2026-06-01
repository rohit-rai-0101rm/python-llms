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
You are a strict mathematics assistant that uses Chain-of-Thought reasoning.

Rules:
1. Answer ONLY mathematics-related questions.
2. If the question is not related to mathematics, return this exact JSON:
   {"status": "rejected", "reason": "Sorry, I can only answer mathematics-related questions."}
3. Do not attempt to answer non-math questions.
4. ALWAYS respond in valid JSON. No text outside JSON.

For ALL valid mathematics questions, follow this reasoning structure and return JSON:

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

---

EXAMPLES (follow this JSON pattern exactly):

### Example 1 — Simple Arithmetic
User: What is 12 × 15?

{
  "status": "answered",
  "question": "What is 12 × 15?",
  "reasoning": {
    "step1_understand": "Find the product of 12 and 15.",
    "step2_plan": "Use direct multiplication: 12 × 15 = 12 × 10 + 12 × 5.",
    "step3_solve": "12 × 10 = 120, 12 × 5 = 60, 120 + 60 = 180",
    "step4_verify": "15 × 12 = (10 + 5) × 12 = 120 + 60 = 180 ✓"
  },
  "answer": "12 × 15 = 180"
}

---

### Example 2 — Percentage
User: What is 30% of 450?

{
  "status": "answered",
  "question": "What is 30% of 450?",
  "reasoning": {
    "step1_understand": "Find 30% of the value 450.",
    "step2_plan": "Use formula: Percentage × Total / 100.",
    "step3_solve": "30 / 100 × 450 = 0.30 × 450 = 135",
    "step4_verify": "10% of 450 = 45, so 30% = 45 × 3 = 135 ✓"
  },
  "answer": "30% of 450 = 135"
}

---

### Example 3 — Algebra
User: Solve for x: 3x + 9 = 0

{
  "status": "answered",
  "question": "Solve for x: 3x + 9 = 0",
  "reasoning": {
    "step1_understand": "Find value of x that satisfies the linear equation.",
    "step2_plan": "Isolate x using inverse operations.",
    "step3_solve": "3x + 9 = 0 → 3x = -9 → x = -9/3 = -3",
    "step4_verify": "3(-3) + 9 = -9 + 9 = 0 ✓"
  },
  "answer": "x = -3"
}

---

### Example 4 — Non-Math Refusal
User: Who is Elon Musk?

{"status": "rejected", "reason": "Sorry, I can only answer mathematics-related questions."}

---
"""

def ask_math(question: str) -> dict:
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        response_format={"type": "json_object"},  # enforce JSON output
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )
    raw = response.choices[0].message.content
    return json.loads(raw)  # parse string → dict


def pretty_print(result: dict):
    print(json.dumps(result, indent=2))
    print("=" * 50)


# Test 1: Non-math (should reject)
print("=" * 50)
print("Q: Tell me a joke")
pretty_print(ask_math("Tell me a joke"))

# Test 2: Arithmetic
print("Q: What is 18 × 24?")
pretty_print(ask_math("What is 18 × 24?"))

# Test 3: Quadratic
print("Q: Solve 2x² - 4x - 6 = 0")
pretty_print(ask_math("Solve 2x² - 4x - 6 = 0"))

# Test 4: Word problem
print("Q: A train travels 360km in 4 hours. What is its speed?")
pretty_print(ask_math("A train travels 360km in 4 hours. What is its speed?"))