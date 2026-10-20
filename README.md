# Competitor Analysis Agent

A concise competitive-analysis agent built with LangGraph and a local Qwen model served by Ollama. The workflow makes one model request and does not require a hosted API key.

## Requirements

- Python 3.10 or later
- [Ollama for Windows](https://ollama.com/download/windows)

## Setup (PowerShell)

From this repository directory:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull qwen3:1.7b
```

Ollama Desktop normally starts its local server automatically. If it is not running, start `ollama serve` in a separate PowerShell window and leave it open.

## Run

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" .\agent.py --company "Linktree" --industry "link-in-bio platform"
```

Replace the company and industry arguments as needed. The model uses general knowledge only; verify current competitor features and market details before relying on them.

## Output

- Five likely competitors and brief differentiators
- Two market opportunities
- Three short recommendations
