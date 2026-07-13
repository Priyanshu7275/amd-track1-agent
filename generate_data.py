"""
Expand 180 hand-written seed examples into ~5,500 synthetic ones.

IMPORTANT: Fireworks is used here for *offline data generation only*.
The deployed agent makes zero Fireworks calls at inference time — this
script runs once, on a development machine, before training.

The generating model is gpt-oss-120b. High temperature (0.9) is used
deliberately: the goal is coverage of many phrasings, not a single
canonical answer.
"""
import os
import json
import time
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["FIREWORKS_API_KEY"],
    base_url="https://api.fireworks.ai/inference/v1",
)
MODEL = "accounts/fireworks/models/gpt-oss-120b"

CATEGORIES = {
    "factual":       "Generate 25 unique factual QA pairs covering capitals, scientists, dates, geography, and science. Each answer 1-2 sentences.",
    "math":          "Generate 25 unique math word problems. Cover percentages, discounts, speed/distance, and arithmetic. The answer must be JUST the number.",
    "sentiment":     "Generate 25 sentiment prompts: 6 positive, 6 negative, 6 neutral, 7 mixed. Answer with exactly one word.",
    "summarization": "Generate 20 summarization tasks: a 3-5 sentence passage plus a one-sentence summary.",
    "ner":           "Generate 25 NER tasks. Extract PERSON, ORG, GPE, and DATE entities.",
    "code_debug":    "Generate 20 Python debugging tasks: a buggy 1-3 line function plus the fix.",
    "code_gen":      "Generate 20 Python function-writing tasks with clear, testable behaviour.",
    "logic":         "Generate 20 constraint-satisfaction puzzles: three people, three items, two constraints, one question.",
}

TARGET_PER_CATEGORY = 500

all_examples = []

for category, instruction in CATEGORIES.items():
    print(f"\n=== {category} ===")
    collected = 0

    for batch in range(40):
        if collected >= TARGET_PER_CATEGORY:
            break

        prompt = instruction + ' Output ONLY JSON lines: {"prompt":"...","answer":"..."}'
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Output JSON lines only. No preamble."},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=4000,
                temperature=0.9,        # variety matters more than precision here
            )

            for line in resp.choices[0].message.content.strip().split("\n"):
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    ex = json.loads(line)
                    if "prompt" in ex and "answer" in ex:
                        ex["category"] = category
                        all_examples.append(ex)
                        collected += 1
                except json.JSONDecodeError:
                    continue    # the model occasionally emits malformed lines

            print(f"  batch {batch}: {collected} total")

        except Exception as e:
            print(f"  batch {batch} failed: {e}")

        time.sleep(0.3)             # stay under the rate limit

# ---- deduplicate on prompt text ---------------------------------------------
seen = set()
unique = []
for ex in all_examples:
    key = ex["prompt"].strip().lower()
    if key not in seen:
        seen.add(key)
        unique.append(ex)

print(f"\nGenerated {len(all_examples)}, {len(unique)} unique after dedup")

with open("/workspace/amd-track1-finetune/synthetic_examples.json", "w") as f:
    json.dump(unique, f, indent=2)
