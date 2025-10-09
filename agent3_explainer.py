# agent3_explainer.py

import json
import os
import sys

# -----------------------------
# Step 0 — Import dependencies
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
except ModuleNotFoundError:
    print("❌ Error: Required package not installed. Run:")
    print("   pip install langchain-openai langchain")
    sys.exit(1)

# -----------------------------
# Step 1 — Load OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = input("Enter your OpenAI API key: ").strip()
    if not api_key:
        print("❌ OpenAI API key is required.")
        sys.exit(1)

# -----------------------------
# Step 2 — Load problem data
data_path = r"E:\CodeAI-DSA\data\problem.json"

if not os.path.exists(data_path):
    print(f"❌ Problem file not found: {data_path}")
    sys.exit(1)

with open(data_path, "r", encoding="utf-8") as f:
    problem_data = json.load(f)

print("📂 Loaded generated code for explanation.")

# -----------------------------
# Step 3 — Initialize model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key="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",
    temperature=0.0
)

# -----------------------------
# Step 4 — Prepare prompt
prompt_text = """
You are an AI tutor who explains code simply and clearly.

Here is the problem description:
{problem_text}

Here is the generated code:
{generated_code}

Please explain:
1. What is the main logic?
2. Step-by-step explanation of each part (loops, functions, variables).
3. Why this approach is efficient.
4. What is the time complexity?
5. What is the space complexity?
6. Mention if there are any better ways to solve it.

Explain it like you’re teaching a beginner.
"""

prompt = PromptTemplate(
    input_variables=["problem_text", "generated_code"],
    template=prompt_text
)

formatted_prompt = prompt.format(
    problem_text=problem_data.get("problem_text", "No description provided."),
    generated_code=problem_data.get("generated_code", "No generated code found.")
)

# -----------------------------
# Step 5 — Generate explanation
try:
    response = llm.invoke(formatted_prompt)
    explanation = getattr(response, "content", None) or getattr(response, "text", None)
    if not explanation:
        raise ValueError("LLM returned empty explanation.")
except Exception as e:
    print(f"❌ Error generating explanation: {e}")
    sys.exit(1)

# Preview explanation
print("\n🧠 Preview of generated explanation:\n")
print(explanation[:400] + ("\n..." if len(explanation) > 400 else ""))

# -----------------------------
# Step 6 — Save to JSON
problem_data["explanation"] = explanation

with open(data_path, "w", encoding="utf-8") as f:
    json.dump(problem_data, f, indent=4, ensure_ascii=False)

print(f"\n✅ Code explanation saved successfully at:\n   {data_path}")

# -----------------------------
# Step 7 — Display final explanation
print("\n--- AI Code Explanation ---\n")
print(explanation)
