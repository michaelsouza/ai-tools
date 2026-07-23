# AI Tools — Quick Guide

## System
- Ubuntu (WSL)

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

## Configure
Create a `.env` file in the repo root:
```bash
MISTRAL_API_KEY=your_mistral_api_key
```

## User options
When suggesting actions to the user, consider offering a clarifying list questions with options (items) to choose from. Also, give your preferences. This helps keep things organized and easy to follow.

## Skill Guidelines & Consistency Check
When creating, refactoring, or reviewing any skill in this repository, always read and validate its consistency against [docs/skills_best_practices_and_evals.md](docs/skills_best_practices_and_evals.md). Ensure that:
- Frontmatter descriptions include clear positive and negative trigger conditions.
- Instructions use imperative directives, focus on objectives and constraints, and avoid rigid scripts or empty "no-ops".
- Skill files remain concise (< 500 lines) and follow the checklist in the guide.