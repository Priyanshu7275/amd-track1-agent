# TokenSmart Router

### A `ZERO_API_CALLS` routing agent for the AMD Developer Hackathon: ACT II

<br>

|                | |
| -------------- | ------------------------------------------------------------ |
| **Track**      | 1 — Hybrid Token-Efficient Routing Agent                      |
| **Author**     | Priyanshu Ranjan                                              |
| **Team**       | BinaryBrains (Team-3516)                                      |
| **Repository** | github.com/Priyanshu7275/amd-track1-agent                     |
| **Container**  | `ghcr.io/priyanshu7275/track1-agent:latest`                   |
| **Model**      | huggingface.co/priyanshu941/track1-qwen-finetuned             |

<br>

---

## The Problem

Track 1 asks for an agent that answers tasks across **eight capability
categories** — factual knowledge, mathematical reasoning, sentiment
classification, summarisation, named entity recognition, code debugging,
logical deduction, and code generation.

The scoring rule is what makes it interesting:

```
  ┌────────────────────────────────────────────────────────┐
  │  1.  ACCURACY GATE                                     │
  │      An LLM judge scores every answer.                 │
  │      Below threshold → excluded from the leaderboard.  │
  │                                                        │
  │  2.  TOKEN EFFICIENCY                                  │
  │      Survivors ranked ASCENDING by tokens spent        │
  │      through the Fireworks proxy. Fewer = better.      │
  └────────────────────────────────────────────────────────┘
```

And buried in the rules, the sentence the whole design turns on:

> *Local model inference inside the container is permitted and counts toward
> accuracy, but **not toward the token score**.*

<br>

---

## The Insight

The obvious move is to route each task to the *cheapest adequate* Fireworks
model. But if local computation is free, the sharper question is:

> ### How many tasks can we answer without calling Fireworks **at all**?

The answer turned out to be *all of them*.

**Zero Fireworks API calls. Zero tokens. Every run.**

The Fireworks fallback was ultimately **removed entirely**. Zero tokens is the
ranking mechanism — a single API call drops you out of the zero-token tier,
while a single empty answer costs about five percent of accuracy. The trade is
not close.

<br>

---

## Architecture — A Three-Layer Cost Ladder

Each task descends the ladder and stops at the first layer that can answer it.

```
                        ┌───────────────────────┐
       task  ──────────▶│  L0   Classifier      │
                        │  regex → 1 of 8 cats  │
                        └───────────┬───────────┘        cost: 0
                                    │
                        ┌───────────▼───────────┐
                        │  L1   Solvers         │
                        │  SymPy · CSP · spaCy  │
                        │  VADER · facts.json   │
                        └───────────┬───────────┘        cost: 0 tokens
                                    │  (no match)
                        ┌───────────▼───────────┐
                        │  L2   Local LLM       │
                        │  Qwen2.5-1.5B         │
                        │  LoRA fine-tuned      │
                        │  ├─ code? → EXECUTE   │
                        │  └─ fail?  → RETRY ×2 │
                        └───────────┬───────────┘        cost: 0 tokens
                                    │  (empty)
                        ┌───────────▼───────────┐
                        │  L3   Local retry     │
                        │  Fireworks: NEVER     │
                        └───────────────────────┘        cost: 0 tokens
```

<br>

### Layer 1 — Deterministic Solvers

> **Never let a probabilistic system decide something a deterministic system
> can prove.**

A 1.5B model performs arithmetic the way it performs everything else — by
pattern completion. Asked for `45 × 2.5 + 60 × 1`, it produces a
*plausible-looking* number. Often the wrong one.

SymPy does not guess.

| Category      | Technique |
| ------------- | --------- |
| **math**      | SymPy — multi-segment travel, nested discounts, sequential percentages, linear equations, unit conversion |
| **logic**     | `python-constraint` CSP — provably correct when a unique solution exists |
| **NER**       | **spaCy** — a trained model, not regex |
| **sentiment** | **VADER** — a 7,500-word scored lexicon with negation and intensifier handling |
| **factual**   | **facts.json** — 3,330 pre-generated lookups, consulted before the model |

<br>

### The Lesson That Shaped Layer 1

For the first day, every solver was hand-written regex over hand-maintained
word lists. And every new test set found exactly one more missing word.

> `"appalling"` wasn't in `_NEG_WORDS`. Add it.
> Next set: `"deafening"`. Add it.
> Next set: `"itchy"`. Add it.
> Next set: `"sluggish"`. Add it.

That is an infinite treadmill. The space of English adjectives is not
enumerable by hand at four in the morning.

**The fix was to stop writing lists and start importing them.**

| Was | Became | Effect |
| --- | ------ | ------ |
| `_POS_WORDS` / `_NEG_WORDS` (~90 words, hand-written) | **VADER** (~7,500 scored words) | Also handles negation (`"not bad"` → positive), intensifiers (`"really awful"`), capitalisation (`"TERRIBLE"`), punctuation (`"awful!!!"`) |
| `_KNOWN_ORGS` / `_KNOWN_PLACES` (~200 entries, hand-written) | **spaCy** `en_core_web_sm` | Handles hyphenated surnames (`Chien-Shiung Wu`), multi-word organisations, and names never seen before |
| `_FACTUAL_QA` (~70 regex patterns, hand-written) | **facts.json** (3,330 entries, generated) | The model is never *asked* a fact it might hallucinate |
| digits only | **word2number** | `"twenty-eight"` → `28` |
| — | **pint** | unit conversion for free |

Every one of these is CPU-only, offline, and costs **zero tokens**.

<br>

### Layer 2 — Bundled Local LLM

**Qwen2.5-1.5B-Instruct**, LoRA fine-tuned on 5,657 category-specific
examples, quantised to 4-bit GGUF (**941 MB**), running on CPU inside the
container via `llama-cpp-python`.

<br>

### The Code Execution Verifier

The single highest-value component in the system.

```
     LLM generates code
             │
             ▼
    ┌─────────────────────┐
    │  extract signature  │    def reverse_string(s): ...
    └──────────┬──────────┘
               ▼
    ┌─────────────────────┐
    │  derive a test      │    assert reverse_string('hello') == 'olleh'
    └──────────┬──────────┘
               ▼
    ┌─────────────────────┐
    │  E X E C U T E      │    subprocess · 3s timeout · sandboxed
    └──────────┬──────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
     PASS ✓        FAIL ✗
        │             │
     return       regenerate ──▶ (up to 2 retries)
```

Code that cannot be **proven to run** is never returned as a confident answer.

This generalises in a way pattern-matching does not: it works on phrasings
never anticipated, because correctness is *tested*, not *assumed*.

<br>

---

## LoRA Fine-Tuning on AMD MI300X

Trained on **AMD Instinct MI300X** via AMD Developer Cloud
(ROCm 7.2 · PyTorch 2.9).

```
    180 hand-written seeds
             │
             │   synthetic expansion  (Fireworks — offline only)
             ▼
    5,557 unique examples
             │
             │   LoRA · r=16 · α=32 · 3 epochs · lr=2e-4 · fp16
             ▼
    loss  1.77 ─────────────────▶  0.28        (13.5 min on MI300X)
             │
             │   merge adapter into base weights
             ▼
    merged model  (3.1 GB fp16)
             │
             │   llama.cpp → GGUF → Q4_K_M
             ▼
    qwen-finetuned-q4_k_m.gguf  (941 MB)   ◀── shipped in the image
```

> **Note:** Fireworks was used **only** to generate training data and the
> facts lookup, offline, before the container was built. The deployed agent
> makes **zero** Fireworks calls at inference time.

**What the fine-tune fixed:** output *shape*. The base model answers a
sentiment task with a paragraph of reasoning. The fine-tuned model answers
with one word.

**What it did not fix:** factual knowledge. It still believes J.J. Thomson
discovered the atomic nucleus, and that Tchaikovsky wrote the Ninth Symphony.
No amount of fine-tuning on five thousand examples teaches a model every fact
that might appear in an evaluation.

That is what `facts.json` is for. **The model stops hallucinating because it
never gets asked.**

<br>

---

## Engineering for 4 GB / 2 vCPU

The grading sandbox is deliberately tight. Every decision follows from it.

| Constraint            | Decision |
| --------------------- | -------- |
| **4 GB RAM**          | Q4_K_M quantisation → 941 MB, leaving ~3 GB headroom |
| **4 GB RAM**          | spaCy `en_core_web_sm` (12 MB) rather than a transformer NER model |
| **2 vCPU**            | `n_threads=2` — matches the allocation exactly |
| **CPU-only**          | `GGML_NATIVE=OFF` — portable across unknown CPUs |
| **10 min ceiling**    | `n_ctx=768` — roughly halves inference time |
| **10 min ceiling**    | Per-category `max_tokens`: 4 for sentiment, 16 for math, 160 for code |
| **10 min ceiling**    | Dynamic budget guard — shortens answers rather than skipping tasks |
| **10 GB image limit** | Final image: **3.2 GB** |
| **60 s startup**      | Cold start: **~5 s** |

<br>

#### The bug that cost twenty points

An early time-budget guard estimated the *future* cost of remaining tasks:

```python
est_needed = remaining_tasks * 20
if budget_left > est_needed:        # ← on 26+ tasks, never true
    answer = local_llm_answer(...)  # ← so this never ran
```

On any task set of 26 or more, the estimate exceeded the budget **from task
one**. The local LLM never fired. Every summarisation and code task returned
an empty string.

**Accuracy: 10.5%.**

The fix was to check only the time actually remaining *right now*:

```python
if budget_left > 30:
    answer = local_llm_answer(...)
```

One line. Accuracy tripled.

<br>

---

## Results

Validated against **ten independently written test sets** the agent had never
seen, run under the exact eval constraints (`--memory=4g --cpus=2`).

| Test set          | Tasks | Correct | Accuracy | Fireworks tokens |
| ----------------- | :---: | :-----: | :------: | :--------------: |
| Practice set      |   8   |    8    | **100%** |      **0**       |
| Hard set A        |  30   |   26    |   87%    |      **0**       |
| Hard set B        |  30   |   28    |   93%    |      **0**       |
| Hard set C        |  30   |   28    |   93%    |      **0**       |
| Hard set D        |  30   |   25    |   85%    |      **0**       |
| Eval simulation ① |  19   |   19    | **100%** |      **0**       |
| Eval simulation ② |  19   |   18    |   95%    |      **0**       |
| Eval simulation ③ |  19   |   18    |   95%    |      **0**       |
| Eval simulation ④ |  19   |   17    |   89%    |      **0**       |
| Eval simulation ⑤ |  19   |   18    |   95%    |      **0**       |

```
  accuracy
     100% ─┤                          ●              ●
           │              ●     ●           ●              ●
      90% ─┤   ●                                 ●
           │                    ●
      80% ─┤
           └───┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬──▶
               A     B     C     D    sim①  sim②  sim③  sim④  sim⑤
```

**Runtime:** ~110 s for 19 tasks — well inside the 600 s ceiling.
**Fireworks tokens:** **zero, on every single run.**

<br>

---

## What I Learned

<br>

> ### ① Deterministic code beats a bigger model.

When accuracy was stuck at 24%, the instinct was to train something larger.
The actual fix was to stop letting a 1.5B model do arithmetic and let SymPy
do it instead. That single change took accuracy from **24% → 90%**.

Every task a solver handles is a task the model *cannot get wrong*.

<br>

> ### ② Import the lexicon; don't write it.

Every test set found one more missing word. `"appalling"`. `"deafening"`.
`"itchy"`. `"sluggish"`. That is not a bug list — it is a treadmill.

Replacing hand-written word lists with **VADER**, **spaCy**, and a generated
**facts.json** ended it permanently. A library that has already solved your
problem is worth more than another hour of your own regex.

<br>

> ### ③ Diagnose before you fix.

Early on I applied speculative patches without knowing *why* the score was
low. It turned out to be a one-line time-budget bug. One honest diagnostic
run would have found it in minutes. I lost hours instead.

<br>

> ### ④ Verification generalises; pattern-matching does not.

Executing generated code proves correctness for phrasings I never
anticipated. Adding another regex only handles the phrasing I just saw.

<br>

> ### ⑤ A failed experiment is still a result.

I implemented **self-consistency sampling** — run the model three times at
temperature 0.7, take the majority vote — because the top-ranked teams
appeared to use it.

Result: **zero improvements, one regression, 33% slower.**

The reason is instructive. Consensus corrects *random* error. My failures
were *systematic* — the model doesn't *sometimes* think Thomson discovered
the nucleus, it consistently does. Sampling three times returns the same
wrong answer three times, while injecting noise into tasks that were already
correct.

I reverted it. Knowing that is worth as much as the technique would have been.

<br>

---

## Repository Layout

```
  README.md               ← this document
  architecture.md         ← detailed design notes
  agent.py                ← the router
  facts.json              ← 3,330 generated knowledge lookups
  Dockerfile
  requirements.txt

  training/
    seed_data.py          ← 180 hand-written examples
    generate_data.py      ← synthetic expansion → 5,557
    prepare_dataset.py    ← chat-template formatting
    train_lora.py         ← LoRA training on MI300X
    merge_lora.py         ← adapter merge
    convert_to_gguf.sh    ← GGUF + Q4_K_M quantisation
    gen_facts.py          ← facts.json generation
```

<br>

---

<div align="center">

**Stack**

`Python 3.11` · `Docker (linux/amd64)` · `Qwen2.5-1.5B (LoRA)` · `llama.cpp`

`SymPy` · `python-constraint` · `spaCy` · `VADER` · `word2number` · `pint`

`PEFT` · `TRL` · `AMD ROCm` · `AMD Instinct MI300X` · `Fireworks AI` · `HuggingFace`

</div>
