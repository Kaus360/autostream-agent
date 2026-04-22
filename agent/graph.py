"""
agent/graph.py
Defines the AgentState TypedDict and builds the LangGraph StateGraph
that drives the full conversation flow for the AutoStream agent.

Flow:
  classify_intent
      ├─ greeting       → rag_response (friendly reply, no RAG lookup)
      ├─ product_inquiry → rag_response (RAG-backed answer)
      └─ high_intent    → lead_collection → lead_capture_tool (when complete)
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from agent.nodes import classify_intent_node, rag_response_node, lead_collection_node, lead_capture_node


# ── Shared conversation state ──────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: list          # Full conversation history as plain dicts {role, content}
    intent: str             # Last classified intent
    name: Optional[str]     # Collected lead field
    email: Optional[str]    # Collected lead field
    platform: Optional[str] # Collected lead field (e.g. YouTube, TikTok …)
    lead_captured: bool     # True once mock_lead_capture() has fired


# ── Routing logic ──────────────────────────────────────────────────────────────

def route_after_classify(state: AgentState) -> str:
    """
    Conditional edge: after classify_intent runs, decide which node comes next.
    high_intent always goes to lead_collection regardless of partial-capture state.
    """
    intent = state.get("intent", "product_inquiry")
    if intent == "high_intent":
        return "lead_collection"
    # Both 'greeting' and 'product_inquiry' are served via rag_response
    return "rag_response"


def route_after_lead_collection(state: AgentState) -> str:
    """
    Conditional edge: after lead_collection runs, check whether all three
    fields are present. Only then proceed to the capture node.
    """
    if state.get("name") and state.get("email") and state.get("platform"):
        return "lead_capture_tool"
    return END


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.

    Returns a compiled graph that accepts an AgentState dict and produces
    an updated AgentState dict after each invocation.
    """
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("classify_intent",  classify_intent_node)
    builder.add_node("rag_response",     rag_response_node)
    builder.add_node("lead_collection",  lead_collection_node)
    builder.add_node("lead_capture_tool", lead_capture_node)

    # Entry point
    builder.set_entry_point("classify_intent")

    # classify_intent → dynamic branch
    builder.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "rag_response":    "rag_response",
            "lead_collection": "lead_collection",
        },
    )

    # rag_response always ends the turn
    builder.add_edge("rag_response", END)

    # lead_collection → either capture or end (waiting for more fields)
    builder.add_conditional_edges(
        "lead_collection",
        route_after_lead_collection,
        {
            "lead_capture_tool": "lead_capture_tool",
            END: END,
        },
    )

    # lead_capture_tool always ends the turn
    builder.add_edge("lead_capture_tool", END)

    return builder.compile()
