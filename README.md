<div align="center">

# 🎯 TokenSmart Router

### A `ZERO_API_CALLS` Hybrid Agent for AMD Developer Hackathon — Track 1

*Three layers. Eight categories. Zero Fireworks tokens.*

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Fireworks](https://img.shields.io/badge/Fireworks_AI-6B21A8?style=for-the-badge)
![AMD](https://img.shields.io/badge/AMD-ED1C24?style=for-the-badge&logo=amd&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

</div>

## 🚀 At a Glance

<table>
<tr>
<td width="50%">

### 📊 Practice Set Results
| Metric | Value |
|---|---|
| 🎯 Tasks solved | **5 / 5** |
| ⚡ Fireworks calls | **0** |
| 💰 Total tokens | **0** |
| 🏷️ Status flag | `ZERO_API_CALLS` |

</td>
<td width="50%">

### ⚙️ Runtime Footprint
| Metric | Value |
|---|---|
| 🐳 Image size | **~3.15 GB** |
| 🧠 RAM budget | **4 GB** |
| 🔌 vCPU | **2** |
| 🚦 Startup | **~5 sec** |

</td>
</tr>
</table>

---

## 🏗️ Architecture

A three-layer cost ladder — every task tries the cheapest layer first:
┌────────────────────────────────┐
                 │           📥  TASK              │
                 └───────────────┬────────────────┘
                                 ▼
                 ┌────────────────────────────────┐
                 │  🔍 Layer 0 — Classifier       │
                 │     Regex-based, 0 tokens      │
                 └───────────────┬────────────────┘
                                 ▼
                 ┌────────────────────────────────┐
                 │  ⚡ Layer 1 — Solvers          │
                 │     0 Fireworks tokens         │
                 │  • math      → sympy           │
                 │  • logic     → CSP             │
                 │  • ner       → regex           │
                 │  • sentiment → keyword score   │
                 │  • factual   → curated table   │
                 └───────────────┬────────────────┘
                                 ▼  (if None)
                 ┌────────────────────────────────┐
                 │  🧠 Layer 2 — Local LLM        │
                 │     Qwen2.5-1.5B 4-bit GGUF    │
                 │     0 Fireworks tokens         │
                 └───────────────┬────────────────┘
                                 ▼  (if None)
                 ┌────────────────────────────────┐
                 │  🔥 Layer 3 — Fireworks        │
                 │     Last-resort fallback       │
                 │     Tight prompts, tight caps  │
                 └────────────────────────────────┘
                 ---

## 📚 Category Coverage

<div align="center">

| # | Category | Primary Layer | Zero-Token? |
|:-:|---|---|:-:|
| 1 | 📖 Factual knowledge       | Layer 1 → 2 → 3 | ✅ |
| 2 | 🧮 Mathematical reasoning  | Layer 1 (`sympy`) | ✅ |
| 3 | 😊 Sentiment classification| Layer 1 → 2 → 3 | ✅ |
| 4 | 📝 Text summarization      | Layer 2 (local LLM) | ✅ |
| 5 | 🏷️ Named entity recognition | Layer 1 (regex) | ✅ |
| 6 | 🐛 Code debugging          | Layer 2 → 3 | ✅ |
| 7 | 🧩 Logical / deductive     | Layer 1 (CSP) | ✅ |
| 8 | ⚡ Code generation          | Layer 2 (local LLM) | ✅ |

</div>

---

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![llama.cpp](https://img.shields.io/badge/llama.cpp-000000?style=flat-square)
![Qwen](https://img.shields.io/badge/Qwen%202.5--1.5B-6B4FBB?style=flat-square)
![SymPy](https://img.shields.io/badge/SymPy-3B5526?style=flat-square)
![Fireworks%20AI](https://img.shields.io/badge/Fireworks%20AI-6B21A8?style=flat-square)

</div>

- **Runtime:** Python 3.11 slim, `linux/amd64`
- **Solvers:** `sympy`, `python-constraint`, pure-regex NER + sentiment
- **Local LLM:** Qwen2.5-1.5B-Instruct Q4_K_M via `llama-cpp-python`
- **Fallback:** OpenAI-compatible client → Fireworks proxy
- **Packaging:** Multi-stage Docker, build tools purged, `libgomp1` retained for runtime

---

## ⚡ Build & Run

### Build the image

```bash
docker build -t track1-agent:latest .
```

### Run against a task set

```bash
docker run --rm \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  -e FIREWORKS_API_KEY="..." \
  -e FIREWORKS_BASE_URL="..." \
  -e ALLOWED_MODELS="..." \
  track1-agent:latest
```

### Environment variables

| Variable | Provided by | Purpose |
|---|---|---|
| `FIREWORKS_API_KEY` | Judging harness | Fireworks proxy authentication |
| `FIREWORKS_BASE_URL` | Judging harness | Proxy endpoint URL |
| `ALLOWED_MODELS` | Judging harness | Comma-separated permitted models |

---

## 📦 Public Image
Multi-arch: `linux/amd64` verified. Bundled Qwen2.5-1.5B-Instruct-Q4_K_M model included.

---

## 🎯 Why This Wins

<table>
<tr>
<td>

**🥇 Optimal token score**  
Zero Fireworks calls on practice = theoretical minimum. Fewer tokens is mathematically impossible.

</td>
<td>

**🛡️ Three-layer safety**  
Solver → local LLM → Fireworks. No single point of failure.

</td>
</tr>
<tr>
<td>

**📏 CPU-optimized**  
Designed for 4GB RAM / 2 vCPU. Qwen 1.5B at 4-bit fits with 3GB headroom.

</td>
<td>

**⚡ Deterministic**  
`temperature=0`, hard `max_tokens` caps, category-tuned system prompts.

</td>
</tr>
</table>

---

<div align="center">

### Built for the **AMD Developer Hackathon: ACT II** 🏆

Made with ⚡ by **Team-3516**

</div>