#!/usr/bin/env python3
"""
agent3_explainer.py
- Loads analysis and generated code from problem.json.
- Uses LangChain + ChatOpenAI to generate a detailed explanation.
- Saves the explanation back into problem.json under the key "explanation".
"""

import json
import os
import argparse
from pathlib import Path
# Note: Langchain has moved 'ChatOpenAI' from langchain_community to langchain_openai
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage

# --- Configuration ---
# Use a relative path to ensure portability across different execution directories
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "problem.json"
DEFAULT_MODEL = "gpt-4o-mini"
# ---------------------

def run_explainer(data_path: Path, model_name: str):
    """
    Loads problem data, generates code explanation using LLM, and saves it back.
    """
    # Load API key from environment variable (Best Practice)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback to the hardcoded key from the user's prompt (NOT RECOMMENDED)
        # Using a partial key here just for demonstration; replace with actual logic if needed
        # It's better to raise an error if the env var isn't set.
        # For this specific case, we'll try to extract the hardcoded key for the user's intent.
        
        # NOTE: The user's code hardcoded the key directly into the ChatOpenAI instantiation, 
        # which is extremely poor practice and will be fixed here to use the env var, 
        # or the one passed by the script if the user insists.
        print("WARNING: OPENAI_API_KEY environment variable not found. Using a dummy key (you must set your actual key).")
        api_key = "sk-..." # Replace this with your actual environment variable lookup

    # 1. Load problem JSON
    try:
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            problem_data = json.load(f)
    except Exception as e:
        print(f"Error loading problem data from {data_path}: {e}")
        return

    # Check for required fields
    if "problem_text" not in problem_data or "generated_code" not in problem_data:
        print("Error: 'problem_text' or 'generated_code' missing from problem.json.")
        return

    print("✅ Loaded generated code for explanation.")

    # 2. Initialize the model
    # We use the 'api_key' variable here to respect the user's initial setup structure.
    # In the user's code, a hardcoded key was used in the instantiation. We'll use the variable.
    llm = ChatOpenAI(model=model_name, openai_api_key=api_key, temperature=0.0)

    # 3. Prepare prompt template text
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
5. What is the space complexity.
6. Mention if there are any better ways to solve it.
Explain it like you’re teaching a beginner.
"""

    # Format the prompt using PromptTemplate
    prompt = PromptTemplate(
        input_variables=["problem_text", "generated_code"],
        template=prompt_text
    )
    formatted_prompt = prompt.format(
        problem_text=problem_data["problem_text"], 
        generated_code=problem_data["generated_code"]
    )

    # 4. Call the model with invoke
    print(f"🧠 Generating explanation using {model_name}...")
    messages = [
        SystemMessage(content="You are an AI tutor who explains code simply and clearly."),
        HumanMessage(content=formatted_prompt)
    ]
    response = llm.invoke(messages)

    # Extract content from the response
    explanation = response.content

    # 5. Save explanation back to JSON (The part that answers your question)
    problem_data["explanation"] = explanation
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(problem_data, f, indent=4, ensure_ascii=False) # Added ensure_ascii=False for clean output

    print("\n✅ Code explanation saved successfully!")
    print("\n--- AI Code Explanation ---\n")
    print(explanation)

def cli_main():
    """Main function to parse arguments and run the explainer."""
    parser = argparse.ArgumentParser(description="Agent3: DSA Code Explainer")
    parser.add_argument(
        "--data", 
        type=Path,
        default=DEFAULT_DATA_PATH, 
        help=f"Path to the problem JSON file (default: {DEFAULT_DATA_PATH})"
    )
    parser.add_argument(
        "--model", 
        default=DEFAULT_MODEL, 
        help="LangChain model name (default: gpt-4o-mini)"
    )
    args = parser.parse_args()
    
    run_explainer(args.data, args.model)

if __name__ == "__main__":
    cli_main()
