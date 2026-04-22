"""
main.py
Entry point for the AutoStream conversational AI agent.

Run with:
    python main.py

The CLI maintains full conversation state across turns using the LangGraph
StateGraph. Type 'quit', 'exit', or 'bye' to end the session.
"""

import os
from dotenv import load_dotenv

# ── Suppress HuggingFace / Deep Learning Logs & Warnings ─────────────────────
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from agent.graph import build_graph, AgentState

load_dotenv()

# ── Validate environment ───────────────────────────────────────────────────────
if not os.getenv("GROQ_API_KEY"):
    raise EnvironmentError(
        "GROQ_API_KEY is not set. "
        "Copy .env.example → .env and add your key."
    )

# ── Banner ─────────────────────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════╗
║        AutoStream AI Agent — powered by ServiceHive  ║
║        Type 'quit' or 'exit' to end the session      ║
╚══════════════════════════════════════════════════════╝
"""


def main():
    print(BANNER)

    # Build and compile the LangGraph graph once
    graph = build_graph()

    # Initialise state — persists across all turns in this session
    state: AgentState = {
        "messages":      [],
        "intent":        "",
        "in_lead_flow":  False,
        "name":          None,
        "email":         None,
        "platform":      None,
        "lead_captured": False,
    }

    while True:
        # ── Get user input ─────────────────────────────────────────────────────
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "bye"}:
            print("\nAgent: Thanks for chatting! Have a great day. 👋\n")
            break

        # ── Append user message to history ─────────────────────────────────────
        state["messages"].append({"role": "user", "content": user_input})

        # ── Run one turn through the graph ─────────────────────────────────────
        try:
            state = graph.invoke(state)
        except Exception as exc:
            print("Agent: Sorry, something went wrong on my end. Please try again.")
            continue

        # ── Print the latest assistant reply ───────────────────────────────────
        agent_reply = ""
        for msg in reversed(state.get("messages", [])):
            if msg.get("role") == "assistant":
                agent_reply = msg.get("content", "")
                break

        if agent_reply:
            print(f"\nAgent: {agent_reply}")

        # ── Stop if lead is fully captured ─────────────────────────────────────
        if state.get("lead_captured"):
            break


if __name__ == "__main__":
    main()
