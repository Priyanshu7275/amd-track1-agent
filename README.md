# AMD Track 1 — Token-Efficient Routing Agent

A hybrid agent for the AMD Developer Hackathon Track 1 that answers
natural language tasks across 8 categories while minimizing Fireworks
API token usage.

## Architecture (v1 baseline)
- Reads tasks from `/input/tasks.json`
- Routes each task to a Fireworks model via the harness-provided proxy
- Writes answers to `/output/results.json`

## Build

    docker build -t track1-agent:latest .

## Run

    docker run --rm \
      -v $(pwd)/input:/input \
      -v $(pwd)/output:/output \
      -e FIREWORKS_API_KEY="..." \
      -e FIREWORKS_BASE_URL="..." \
      -e ALLOWED_MODELS="..." \
      track1-agent:latest

## Environment variables (injected by harness)
- `FIREWORKS_API_KEY` — API key for Fireworks proxy
- `FIREWORKS_BASE_URL` — proxy endpoint
- `ALLOWED_MODELS` — comma-separated list of permitted model IDs