#!/usr/bin/env python3
"""
agent1_analyzer.py
- Uses LangChain + ChatOpenAI to analyze a DSA problem statement.
- Produces ../data/problem.json (creates folder if needed).
"""

import os
import json
import argparse
import logging
import langchain_community
# If no error occurs, press Ctrl+Z then Enter (Windows) or Ctrl+D (Linux/macOS) to exit
import re
from pathlib import Path
from typing import Dict, Any

# LangChain imports
# LangChain imports
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain # <--- CORRECTED IMPORT

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


DEFAULT_PROMPT = """
You are a DSA Problem Analyzer. Your job is to read the short problem statement and produce a concise machine-friendly JSON containing:
1) input: short description of the input (data types, formats)
2) output: short description of the expected output
3) constraints: any constraints like time/space complexity expectations, input size limits, value ranges -- if unknown, say "unknown"
4) edge_cases: a short list of likely edge cases (comma separated or array)
Return ONLY valid JSON. Use these keys exactly: input, output, constraints, edge_cases.
Problem: {problem}
"""

def make_chain(model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    prompt = PromptTemplate(input_variables=["problem"], template=DEFAULT_PROMPT)
    return LLMChain(llm=llm, prompt=prompt)


def extract_json_from_text(text: str) -> str:
    """
    Try to extract the first {...} JSON block from LLM text output.
    """
    # Greedy match of the outermost braces (simple heuristic)
    match = re.search(r"\{(?:[^{}]|(?R))*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0)
    # fallback: try to guess by matching lines that look like "key: value"
    return None


def parse_response(raw: str) -> Dict[str, Any]:
    """
    Parse LLM raw text into a dict. If parsing fails, produce a safe fallback.
    """
    # 1) Try direct json.loads
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2) Try to extract JSON substring
    json_candidate = extract_json_from_text(raw)
    if json_candidate:
        try:
            return json.loads(json_candidate)
        except Exception:
            pass

    # 3) Heuristic fallback: try to parse simple "key: value" lines
    data = {"input": "unknown", "output": "unknown", "constraints": "unknown", "edge_cases": []}
    for line in raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip().lower()
            v = v.strip().strip('"- ')
            if "input" in k:
                data["input"] = v
            elif "output" in k:
                data["output"] = v
            elif "constraint" in k:
                data["constraints"] = v
            elif "edge" in k:
                # try to split into list
                parts = re.split(r",|\;|\n", v)
                data["edge_cases"] = [p.strip() for p in parts if p.strip()]
    # attach raw analysis for debugging
    data["_raw_analysis"] = raw.strip()
    return data


def analyze_problem(problem_text: str, model_name: str = "gpt-4o-mini", temperature: float = 0.0) -> Dict[str, Any]:
    chain = make_chain(model_name=model_name, temperature=temperature)
    logging.info("Sending problem to LLM...")
    raw = chain.run(problem=problem_text)
    logging.debug("Raw LLM output:\n%s", raw)
    parsed = parse_response(raw)

    # Normalise types: ensure edge_cases is list
    if isinstance(parsed.get("edge_cases"), str):
        parsed["edge_cases"] = [s.strip() for s in re.split(r",|\n|;", parsed["edge_cases"]) if s.strip()]

    # Ensure keys exist
    for k in ("input", "output", "constraints", "edge_cases"):
        parsed.setdefault(k, "unknown" if k != "edge_cases" else [])

    return parsed


def save_analysis(problem_text: str, analysis: Dict[str, Any], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"problem_text": problem_text}
    data.update(analysis)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging.info("Saved analysis to %s", out_path)


def cli_main():
    parser = argparse.ArgumentParser(description="Agent1: DSA Problem Analyzer")
    parser.add_argument("problem", nargs="?", help="Problem text (wrap in quotes). If omitted, read from stdin.")
    parser.add_argument("--file", help="Read problem from a text file")
    parser.add_argument("--out", default="../data/problem.json", help="Output JSON path")
    parser.add_argument("--model", default="gpt-4o-mini", help="LangChain model name")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    parser.add_argument("--no-save", action="store_true", help="Don't write output file; print to stdout")
    args = parser.parse_args()

    if args.file:
        problem_text = Path(args.file).read_text(encoding="utf-8").strip()
    elif args.problem:
        problem_text = args.problem.strip()
    else:
        logging.info("Reading problem from stdin (end with Ctrl-D).")
        problem_text = ""
        try:
            import sys
            problem_text = sys.stdin.read().strip()
        except Exception:
            pass

    if not problem_text:       
        logging.error("No problem text provided. Exiting.")
        return

    analysis = analyze_problem(problem_text, model_name=args.model, temperature=args.temperature)

    if args.no_save:
        print(json.dumps({"problem_text": problem_text, **analysis}, indent=2, ensure_ascii=False))
    else:
        out_path = Path(args.out)
        save_analysis(problem_text, analysis, out_path)
        print("✅ Problem analyzed and saved to", out_path)


if __name__ == "__main__":
    cli_main()
