"""
Generate a general-knowledge lookup table via Fireworks.

Run once, offline. The resulting facts.json is baked into the Docker image
and consulted before the local LLM — so the model is never asked a factual
question it might hallucinate on.

Zero inference-time tokens: this runs on the dev machine, not in the container.
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

DOMAINS = [
    "world geography: capitals, rivers, mountains, deserts, oceans, islands, lakes",
    "physical science: elements, atomic numbers, chemical symbols, constants, units",
    "astronomy: planets, moons, distances, missions, telescopes",
    "biology: classification, anatomy, famous discoveries, diseases",
    "history: wars, treaties, revolutions, empires, key dates",
    "literature: novels and their authors, poets, playwrights, prizes",
    "art and architecture: paintings, sculptors, movements, landmarks",
    "classical music: composers, symphonies, operas, instruments",
    "inventions and technology: who invented what, and when",
    "medicine: vaccines, discoveries, Nobel laureates",
    "mathematics: theorems, constants, mathematicians",
    "sports: records, championships, notable athletes",
    "world politics: leaders, organisations, founding dates",
    "modern computing: companies, founders, languages, protocols",
]

PROMPT = """Generate 60 general-knowledge facts about {domain}.

Output ONLY JSON lines. Each line:
{{"key": "<distinctive lowercase phrase that would appear in a question>", "answer": "<the answer, under 15 words>"}}

The key must be the phrase a question would contain — NOT the full question.

Examples:
{{"key": "largest island", "answer": "Greenland"}}
{{"key": "ninth symphony", "answer": "Ludwig van Beethoven"}}
{{"key": "circulation of blood", "answer": "William Harvey"}}
{{"key": "atomic number of gold", "answer": "79"}}
{{"key": "deepest lake", "answer": "Lake Baikal, at 1,642 metres"}}

Keys must be specific enough to avoid false matches. No duplicates."""

facts = {}

for domain in DOMAINS:
    print(f"\n=== {domain[:40]} ===")
    for batch in range(6):          # ~360 facts per domain
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Output JSON lines only. No preamble."},
                    {"role": "user", "content": PROMPT.format(domain=domain)},
                ],
                max_tokens=4000,
                temperature=0.8,
            )
            for line in resp.choices[0].message.content.strip().split("\n"):
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    row = json.loads(line)
                    k = row.get("key", "").strip().lower()
                    v = row.get("answer", "").strip()
                    if k and v and len(k) > 4:
                        facts[k] = v
                except json.JSONDecodeError:
                    continue
            print(f"  batch {batch}: {len(facts)} total")
        except Exception as e:
            print(f"  batch {batch} failed: {e}")
        time.sleep(0.3)

# Longest keys first — "largest island" should not shadow "largest island in asia"
ordered = dict(sorted(facts.items(), key=lambda kv: -len(kv[0])))

with open("facts.json", "w", encoding="utf-8") as f:
    json.dump(ordered, f, indent=1, ensure_ascii=False)

print(f"\nWrote {len(ordered)} facts to facts.json")
