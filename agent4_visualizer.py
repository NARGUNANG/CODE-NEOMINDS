# agent4_visualizer.py

import json
from pathlib import Path
import streamlit as st

# 🟡 Step 1 — Locate and load JSON file
data_path = Path(__file__).parent.parent / "data" / "problem.json"

with open(data_path, "r", encoding="utf-8") as f:
    problem_data = json.load(f)

# 🟡 Step 2 — Extract fields
problem_text = problem_data.get("problem_text", "❌ No problem text found.")
generated_code = problem_data.get("generated_code", "❌ No code found.")
explanation = problem_data.get("explanation", "❌ No explanation found.")

# 🟡 Step 3 — Display using Streamlit
st.title("🧠 CodeAI DSA Visualizer")

st.subheader("📝 Problem Statement")
st.write(problem_text)

st.subheader("💻 Generated Code")
st.code(generated_code, language="python")

st.subheader("📊 Explanation")
st.write(explanation)
