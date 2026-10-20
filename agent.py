"""Generate a concise competitive analysis with one local Qwen inference."""

import argparse
import os
from typing import TypedDict

import requests
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

load_dotenv()


class AnalysisState(TypedDict):
    company: str
    industry: str
    final_report: str


def generate_report(state: AnalysisState) -> AnalysisState:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    response = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": os.getenv("OLLAMA_MODEL", "qwen3:1.7b"),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Give a fast, concise competitive analysis from general knowledge only. "
                        "List exactly 5 likely competitors with a brief differentiator, 2 opportunities, "
                        "and 3 short recommendations. Do not claim live research or give exact current "
                        "prices. Keep under 120 words. Output the report only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Analyze {state['company']} in the {state['industry']} industry. "
                        "Focus on direct alternatives and practical differentiation."
                    ),
                },
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.8,
                "num_ctx": 2048,
                "num_predict": 160,
            },
            "keep_alive": "5m",
        },
        timeout=180,
    )
    response.raise_for_status()
    report = response.json().get("message", {}).get("content", "")
    if not report.strip():
        raise RuntimeError(
            "Ollama returned an empty response. Check that the model is available "
            "with `ollama list`, then retry."
        )
    return {"final_report": report.strip()}


def build_graph():
    graph = StateGraph(AnalysisState)
    graph.add_node("report", generate_report)
    graph.set_entry_point("report")
    graph.add_edge("report", END)
    return graph.compile()


def main():
    parser = argparse.ArgumentParser(description="Competitive Analysis Agent")
    parser.add_argument("--company", default="Notion", help="Company to analyze")
    parser.add_argument("--industry", default="productivity and collaboration software", help="Industry")
    args = parser.parse_args()

    print(
        f"\nAnalyzing {args.company} with local Qwen 1.7B (one model call)...\n",
        flush=True,
    )

    agent = build_graph()
    result = agent.invoke({
        "company": args.company,
        "industry": args.industry,
        "final_report": "",
    })

    print("=" * 60)
    print("📊 COMPETITIVE ANALYSIS REPORT")
    print("=" * 60)
    print(result["final_report"] or "The local model returned an empty report. Try running the command again.")


if __name__ == "__main__":
    main()
