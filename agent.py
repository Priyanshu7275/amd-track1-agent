"""
AMD Track 1 — Hybrid Token-Efficient Routing Agent (v2.1)

Pipeline:
  Task → classify → solver ladder → Fireworks fallback

Zero-Fireworks-token solvers:
  - MathSolver     : sympy + pattern matching
  - LogicSolver    : python-constraint for CSP puzzles
  - NERSolver      : pure regex
  - SentimentSolver: keyword scoring with confidence gate
  - FactualSolver  : lightweight pattern matches

Fireworks fallback: cheapest allowed model, tight per-category prompts,
hard max_tokens caps, temperature=0.
"""
import json
import os
import re
import sys
from typing import Optional

# ---- optional imports so agent still runs if any fail ----
try:
    import sympy
    from sympy import sympify
    _HAVE_SYMPY = True
except Exception:
    _HAVE_SYMPY = False

try:
    from constraint import Problem, AllDifferentConstraint
    _HAVE_CONSTRAINT = True
except Exception:
    _HAVE_CONSTRAINT = False
try:
    from llama_cpp import Llama
    _HAVE_LLAMA = True
except Exception as _e:
    _HAVE_LLAMA = False
    _LLAMA_IMPORT_ERROR = str(_e)
else:
    _LLAMA_IMPORT_ERROR = None

# Path where the model file will live inside the Docker container
_MODEL_PATH = "/app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
_LOCAL_LLM = None  # will be set at startup if model file exists
from openai import OpenAI

INPUT_PATH = "/input/tasks.json"
OUTPUT_PATH = "/output/results.json"
# =========================================================================
# LOCAL LLM (llama.cpp) — zero Fireworks tokens
# =========================================================================
def load_local_llm():
    """Load the local GGUF model if the file and library are available."""
    global _LOCAL_LLM
    if not _HAVE_LLAMA:
        print(f"[local_llm] llama_cpp import failed: {_LLAMA_IMPORT_ERROR}", file=sys.stderr)
        return
    if not os.path.exists(_MODEL_PATH):
        print(f"[local_llm] model file not found at {_MODEL_PATH}, skipping", file=sys.stderr)
        return
    try:
        print(f"[local_llm] loading model from {_MODEL_PATH}", file=sys.stderr)
        _LOCAL_LLM = Llama(
            model_path=_MODEL_PATH,
            n_ctx=2048,
            n_threads=2,       # matches 2 vCPU eval env
            n_batch=256,
            verbose=False,
        )
        print(f"[local_llm] loaded successfully", file=sys.stderr)
    except Exception as e:
        print(f"[local_llm] failed to load: {e}", file=sys.stderr)
        _LOCAL_LLM = None

def local_llm_answer(category: str, prompt: str) -> Optional[str]:
    """Answer with the local model. Returns None if model unavailable or fails."""
    if _LOCAL_LLM is None:
        return None
    try:
        system_prompt = SYSTEM_PROMPTS.get(category, "Answer concisely. No preamble.")
        max_tokens = MAX_TOKENS.get(category, 128)

        # Qwen chat format
        full_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        result = _LOCAL_LLM(
            full_prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            stop=["<|im_end|>", "<|im_start|>"],
            echo=False,
        )
        answer = result["choices"][0]["text"].strip()
        # Strip markdown code fences if the model added them
        answer = re.sub(r"^```(?:python|py|json|)?\s*\n?", "", answer)
        answer = re.sub(r"\n?```\s*$", "", answer)
        answer = answer.strip()
        return answer if answer else None
    except Exception as e:
        print(f"[local_llm] inference failed: {e}", file=sys.stderr)
        return None
# =========================================================================
# CLASSIFIER — regex-based, cheap
# =========================================================================
def classify(prompt: str) -> str:
    p = prompt.lower()

    if re.search(r"\b(sum|total|percent|%|difference|product|divide|multiply|"
                 r"add|subtract|how many|how much|remain|left|average|mean)\b", p) \
       and re.search(r"\d", p):
        return "math"

    if re.search(r"\b(bug|fix|debug|error|corrected|incorrect)\b", p) \
       and ("def " in p or "function" in p or "return" in p):
        return "code_debug"

    if re.search(r"\bwrite (a )?(python )?function\b", p) or \
       re.search(r"\bimplement\b", p):
        return "code_gen"

    if re.search(r"\b(named entit|extract .*(entities|names)|list .*(people|organizations|places))\b", p):
        return "ner"

    if re.search(r"\b(sentiment|positive|negative|neutral|classif)\b", p):
        return "sentiment"

    if re.search(r"\b(each (own|has)|who owns|different (pet|color|job|house)|"
                 r"three (friends|people))\b", p):
        return "logic"

    if re.search(r"\b(summari[sz]e|summary|in one sentence|briefly)\b", p):
        return "summarization"

    return "factual"

# =========================================================================
# MATH SOLVER — zero tokens
# =========================================================================
def solve_math(prompt: str) -> Optional[str]:
    if not _HAVE_SYMPY:
        return None
    try:
        p = prompt.lower()

        # "X items. Sells Y% on Monday and Z more on Tuesday. How many remain?"
        m = re.search(r"(\d+)\s*items?.*sells?\s*(\d+)\s*%.*?(\d+)\s*more", p)
        if m:
            total, pct, more = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return str(total - (total * pct // 100) - more)

        # "X% of Y"
        m = re.search(r"(\d+)\s*%\s+of\s+(\d+)", p)
        if m:
            pct, of = int(m.group(1)), int(m.group(2))
            result = pct * of / 100
            return str(int(result)) if result.is_integer() else str(result)

        # "what is <arithmetic expression>"
        m = re.search(r"what is\s+([\d\+\-\*\/\(\)\.\s]+)\??", p)
        if m:
            expr = m.group(1).strip()
            try:
                return str(sympify(expr))
            except Exception:
                pass

        return None
    except Exception:
        return None

# =========================================================================
# LOGIC SOLVER — CSP puzzles, zero tokens
# =========================================================================
def solve_logic(prompt: str) -> Optional[str]:
    if not _HAVE_CONSTRAINT:
        return None
    try:
        p = prompt.lower()

        m = re.search(r"(\w+),\s*(\w+),?\s*(?:and\s+)?(\w+),?\s*each own[s]? a different\s+"
                      r"\w+:\s*([\w,\s]+)\.\s*(.*?)who owns the (\w+)", p, re.DOTALL)
        if not m:
            return None

        people = [m.group(1), m.group(2), m.group(3)]
        items = [x.strip() for x in m.group(4).split(",") if x.strip()]
        constraints_text = m.group(5)
        query_item = m.group(6).strip()

        if len(items) != len(people):
            return None

        problem = Problem()
        for person in people:
            problem.addVariable(person, items)
        problem.addConstraint(AllDifferentConstraint())

        for person in people:
            m2 = re.search(rf"{person} owns the (\w+)", constraints_text)
            if m2 and m2.group(1) in items:
                target = m2.group(1)
                problem.addConstraint(lambda v, t=target: v == t, [person])
            m3 = re.search(rf"{person} does not own the (\w+)", constraints_text)
            if m3 and m3.group(1) in items:
                target = m3.group(1)
                problem.addConstraint(lambda v, t=target: v != t, [person])

        solutions = problem.getSolutions()
        if len(solutions) == 1:
            for person, item in solutions[0].items():
                if item == query_item:
                    return person.capitalize()
        return None
    except Exception:
        return None

# =========================================================================
# NER SOLVER — pure regex, zero tokens, with diagnostics
# =========================================================================
_MONTHS = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
_ORG_SUFFIXES = r"(?:Inc|LLC|Ltd|Corp|Corporation|Company|Co|GmbH|AG|SA|AI|Labs|Group|University)"

_KNOWN_PLACES = {
    "Berlin", "Paris", "London", "Tokyo", "New York", "San Francisco", "Beijing",
    "Sydney", "Mumbai", "Delhi", "Moscow", "Madrid", "Rome", "Toronto", "Boston",
    "Chicago", "Seattle", "Amsterdam", "Dublin", "Singapore", "Hong Kong",
    "Canberra", "Ottawa", "Washington",
    "USA", "UK", "India", "China", "Japan", "Germany", "France", "Spain",
    "Italy", "Australia", "Canada", "Brazil", "Russia",
}

def solve_ner(prompt: str) -> Optional[str]:
    print(f"[ner] CALLED with prompt: {prompt[:80]!r}", file=sys.stderr)
    try:
        m = re.search(r"(?:from|following)\s*:\s*(.+)$", prompt, re.IGNORECASE | re.DOTALL)
        text = m.group(1).strip() if m else prompt
        print(f"[ner] analyzing text: {text!r}", file=sys.stderr)

        entities = []

        # PERSONS: two consecutive capitalized words
        for match in re.finditer(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", text):
            name = match.group(1)
            if name in _KNOWN_PLACES:
                continue
            entities.append((name, "PERSON"))

        # ORGS
        for match in re.finditer(rf"\b((?:[A-Z][A-Za-z]+ ){{0,3}}{_ORG_SUFFIXES})\b", text):
            entities.append((match.group(1).strip(), "ORG"))

        # PLACES
        for place in _KNOWN_PLACES:
            if re.search(rf"\b{re.escape(place)}\b", text):
                entities.append((place, "GPE"))

        # DATES
        for match in re.finditer(rf"\b(?:last |next |this )?{_MONTHS}\b", text, re.IGNORECASE):
            entities.append((match.group(0), "DATE"))
        for match in re.finditer(r"\b(19|20)\d{2}\b", text):
            entities.append((match.group(0), "DATE"))

        print(f"[ner] found entities: {entities}", file=sys.stderr)

        # Deduplicate preserving order
        seen = set()
        unique = []
        for name, label in entities:
            key = (name.lower(), label)
            if key in seen:
                continue
            seen.add(key)
            unique.append((name, label))

        if not unique:
            print(f"[ner] returning None (no entities found)", file=sys.stderr)
            return None

        result = ", ".join(f"{name} ({label})" for name, label in unique)
        print(f"[ner] returning: {result!r}", file=sys.stderr)
        return result
    except Exception as e:
        print(f"[ner] EXCEPTION: {e}", file=sys.stderr)
        return None

# =========================================================================
# SENTIMENT SOLVER — keyword-scored, confidence-gated
# =========================================================================
_POS_WORDS = {
    "great", "excellent", "amazing", "love", "loved", "wonderful", "perfect",
    "fantastic", "awesome", "best", "good", "brilliant", "outstanding", "happy",
    "delighted", "recommend", "enjoyed", "impressive", "superb", "flawless"
}
_NEG_WORDS = {
    "terrible", "awful", "hate", "hated", "bad", "worst", "poor", "disappointing",
    "broken", "useless", "waste", "annoying", "scratches", "fails", "failed",
    "problem", "issue", "expensive", "slow", "buggy", "crash", "crashed"
}
_MIXED_HINTS = {"but", "however", "although", "though", "yet", "still"}

def solve_sentiment(prompt: str) -> Optional[str]:
    m = re.search(r"(?:review|text|following)\s*:\s*(.+)$", prompt, re.IGNORECASE | re.DOTALL)
    text = (m.group(1) if m else prompt).lower()

    words = set(re.findall(r"\b\w+\b", text))
    pos = len(words & _POS_WORDS)
    neg = len(words & _NEG_WORDS)
    mixed = bool(words & _MIXED_HINTS) and pos > 0 and neg > 0

    if mixed:
        return "mixed"
    if pos >= 2 and neg == 0:
        return "positive"
    if neg >= 2 and pos == 0:
        return "negative"
    if pos == 1 and neg == 0:
        return "positive"
    if neg == 1 and pos == 0:
        return "negative"
    return None

# =========================================================================
# FACTUAL SOLVER — tiny lookup table
# =========================================================================
_CAPITALS = {
    "australia": "Canberra (near Lake Burley Griffin)",
    "france": "Paris (on the River Seine)",
    "japan": "Tokyo (on Tokyo Bay)",
    "germany": "Berlin (on the River Spree)",
    "italy": "Rome (on the River Tiber)",
    "spain": "Madrid (on the River Manzanares)",
    "canada": "Ottawa (on the Ottawa River)",
    "russia": "Moscow (on the Moskva River)",
}

def solve_factual(prompt: str) -> Optional[str]:
    p = prompt.lower()
    m = re.search(r"capital of (\w+)", p)
    if m and m.group(1) in _CAPITALS:
        return _CAPITALS[m.group(1)]
    return None

# =========================================================================
# FIREWORKS FALLBACK — per-category tuned
# =========================================================================
SYSTEM_PROMPTS = {
    "math": "Answer with only the final numeric answer. No explanation.",
    "sentiment": "Reply with exactly one word: positive, negative, or neutral.",
    "ner": "List entities as 'text (TYPE)' separated by commas. Nothing else.",
    "summarization": "Summarize in one concise sentence. No preamble.",
    "code_debug": "Return only the corrected code. No commentary.",
    "code_gen": "Return only the requested code. No commentary.",
    "logic": "Reply with only the answer name. Nothing else.",
    "factual": "Answer directly and concisely. No preamble.",
}

MAX_TOKENS = {
    "math": 32,
    "sentiment": 8,
    "ner": 128,
    "summarization": 96,
    "code_debug": 384,
    "code_gen": 384,
    "logic": 16,
    "factual": 128,
}

def fireworks_answer(client, model: str, category: str, prompt: str) -> str:
    system_prompt = SYSTEM_PROMPTS.get(category, "Answer concisely. No preamble.")
    max_tokens = MAX_TOKENS.get(category, 128)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()

# =========================================================================
# MAIN
# =========================================================================
def main():
    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ["FIREWORKS_BASE_URL"]
    allowed = [m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()]
    if not allowed:
        print("No allowed models available", file=sys.stderr)
        sys.exit(1)
    default_model = allowed[0]

    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"[startup] solvers: sympy={_HAVE_SYMPY}, constraint={_HAVE_CONSTRAINT}", file=sys.stderr)
    print(f"[startup] default_model={default_model}", file=sys.stderr)
    load_local_llm()
    print(f"[startup] local_llm ready: {_LOCAL_LLM is not None}", file=sys.stderr)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    results = []
    for task in tasks:
        tid = task["task_id"]
        prompt = task["prompt"]
        category = classify(prompt)
        answer = None

        # Try zero-token solvers first
        if category == "math":
            answer = solve_math(prompt)
        elif category == "logic":
            answer = solve_logic(prompt)
        elif category == "ner":
            answer = solve_ner(prompt)
        elif category == "sentiment":
            answer = solve_sentiment(prompt)
        elif category == "factual":
            answer = solve_factual(prompt)

        # Fireworks fallback
        # Layer 2: try local model (zero Fireworks tokens)
        if answer is None or answer == "":
            answer = local_llm_answer(category, prompt)
            if answer:
                print(f"[{tid}] local_llm answered", file=sys.stderr)

        # Layer 3: Fireworks fallback (real tokens)
        if answer is None or answer == "":
            try:
                answer = fireworks_answer(client, default_model, category, prompt)
            except Exception as e:
                print(f"[{tid}] fireworks failed: {e}", file=sys.stderr)
                answer = ""

        print(f"[{tid}] category={category} -> {answer[:60]!r}", file=sys.stderr)
        results.append({"task_id": tid, "answer": answer})

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(results)} results", file=sys.stderr)

if __name__ == "__main__":
    main()