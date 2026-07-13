"""
Convert the raw (prompt, answer, category) examples into the chat format
Qwen expects for supervised fine-tuning.

Each example becomes a three-turn conversation:

    system    -> the same terse instruction the agent uses at inference time
    user      -> the task prompt
    assistant -> the gold answer

Using the *identical* system prompt at training and inference time is what
teaches the model to produce correctly-shaped output — one word for
sentiment, a bare number for math, code with no commentary.
"""
import json
from seed_data import SEED_EXAMPLES

# These are byte-for-byte the prompts used in agent.py at inference time.
SYSTEM_PROMPTS = {
    "math":          "Output the number only. Nothing else.",
    "sentiment":     "Output one word: positive, negative, neutral, or mixed.",
    "ner":           "Output only: Entity (TYPE), Entity (TYPE). Types: PERSON, ORG, GPE, DATE.",
    "summarization": "One sentence. No preamble.",
    "code_debug":    "Corrected code only. No explanation.",
    "code_gen":      "Code only. No explanation.",
    "logic":         "Output the answer only. Nothing else.",
    "factual":       "Answer in under 15 words. Direct. No preamble.",
}


def build(examples, out_path):
    """Write examples to a JSONL file in TRL's expected message format."""
    written = 0
    with open(out_path, "w") as f:
        for ex in examples:
            prompt = ex.get("prompt", "")
            answer = ex.get("answer", "")
            category = ex.get("category", "factual")

            # Synthetic generation occasionally emits malformed rows;
            # drop anything that isn't a clean string pair.
            if not isinstance(prompt, str) or not isinstance(answer, str):
                continue
            if not prompt.strip() or not answer.strip():
                continue

            row = {
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPTS[category]},
                    {"role": "user",      "content": prompt.strip()},
                    {"role": "assistant", "content": answer.strip()},
                ]
            }
            f.write(json.dumps(row) + "\n")
            written += 1

    return written


if __name__ == "__main__":
    # In the full pipeline this also merges in synthetic_examples.json,
    # deduplicating on the prompt text. Final corpus: 5,657 examples.
    n = build(SEED_EXAMPLES, "/workspace/amd-track1-finetune/train_data.jsonl")
    print(f"Wrote {n} training examples")
