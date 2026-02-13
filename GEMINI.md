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