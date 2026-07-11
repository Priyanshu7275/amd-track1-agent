import json
import os
import sys
from openai import OpenAI

INPUT_PATH = "/input/tasks.json"
OUTPUT_PATH = "/output/results.json"

# Concise system prompt. Short = fewer input tokens per call.
SYSTEM_PROMPT = (
    "You are a precise assistant. Answer the user's question directly and "
    "concisely. No preamble, no meta-commentary, no chain-of-thought. "
    "Give only the final answer."
)

def load_tasks():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def write_results(results):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def answer_with_fireworks(client, model, prompt):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,   # hard cap output
        temperature=0.0,  # deterministic = reproducible
    )
    return resp.choices[0].message.content.strip()

def main():
    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ["FIREWORKS_BASE_URL"]
    allowed_models = [m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()]

    if not allowed_models:
        print("No allowed models available", file=sys.stderr)
        sys.exit(1)

    # Baseline: use the FIRST allowed model for everything.
    # Assumption: the list is typically ordered cheap→strong. We'll verify.
    default_model = allowed_models[0]

    client = OpenAI(api_key=api_key, base_url=base_url)
    tasks = load_tasks()

    results = []
    for task in tasks:
        task_id = task["task_id"]
        prompt = task["prompt"]
        try:
            answer = answer_with_fireworks(client, default_model, prompt)
        except Exception as e:
            # Never crash the whole run because of one bad task
            print(f"Task {task_id} failed: {e}", file=sys.stderr)
            answer = ""
        results.append({"task_id": task_id, "answer": answer})

    write_results(results)

if __name__ == "__main__":
    main()