# Tagore Songs Auto-Research Agent

You are an autonomous research agent. Read `program.md` for full instructions.

## Quick Start

When the user says "start" or "run the research loop":

1. Read `program.md` for the full protocol
2. Read `research_log.md` to see what's been done
3. Run `uv run python dataset.py` to see the dataset summary
4. Begin the experiment loop described in `program.md`

## Key Rules

- Each experiment is a self-contained script at `experiments/experiment_NNN.py`
- Run scripts with `uv run python experiments/experiment_NNN.py`
- If a script needs a new dependency, add it to `pyproject.toml` and run `uv sync` first
- **NEVER use LLMs or pre-trained language model embeddings** — the data is likely in their training sets
- Always cross-validate ML models. Never report training accuracy.
- Log every experiment to `research_log.md`, including failures
- Use web search to build domain context before experiments
- Keep going indefinitely. Don't stop after a few experiments. Run overnight.

## Working Directory

All paths are relative to `/Users/ppm/code/tagore_auto_analysis/`.
