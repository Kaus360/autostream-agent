"""
agent/nodes.py
All LangGraph node functions. Each function receives the current AgentState,
performs its work, and returns a *partial* state dict with only the keys
it updates. LangGraph merges these updates into the running state.

Nodes:
  classify_intent_node  — classifies user intent via Groq LLM
  rag_response_node     — answers product / greeting queries using RAG + Groq
  lead_collection_node  — sequentially collects name → email → platform
  lead_capture_node     — fires mock_lead_capture() when all fields present
"""

import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage as LCHuman
from dotenv import load_dotenv

from agent.intent import classify_intent
from agent.tools import mock_lead_capture
from rag.retriever import retrieve

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_llm(temperature: float = 0.4) -> ChatGroq:
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature,
    )


def _append_assistant(state: dict, text: str) -> list:
    """Return a new messages list with the assistant reply appended."""
    messages = list(state.get("messages", []))
    messages.append({"role": "assistant", "content": text})
    return messages


def _last_user_message(state: dict) -> str:
    """Extract the most recent user message from the messages list."""
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — classify_intent_node
# ─────────────────────────────────────────────────────────────────────────────

def classify_intent_node(state: dict) -> dict:
    """
    Classifies the latest user message into one of:
      greeting | product_inquiry | high_intent

    Updates:  intent
    """
    user_msg = _last_user_message(state)
    intent = classify_intent(user_msg)
    return {"intent": intent}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — rag_response_node
# ─────────────────────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are AutoStream's friendly and knowledgeable support assistant for ServiceHive.
AutoStream is an AI-powered automated video editing SaaS for content creators.

You have access to retrieved knowledge-base excerpts below. Use them to answer
the user's question accurately and concisely. If a question falls outside the
knowledge base, say so honestly rather than guessing.

Keep answers under 120 words. Be warm, professional, and helpful.

Retrieved Context:
{context}
"""

GREETING_SYSTEM_PROMPT = """You are AutoStream's friendly onboarding assistant for ServiceHive.
AutoStream is an AI-powered automated video editing SaaS for content creators.

Greet the user warmly, introduce AutoStream in one or two sentences, and invite
them to ask about features, pricing, or to get started. Be enthusiastic but concise
(under 60 words).
"""


def rag_response_node(state: dict) -> dict:
    """
    For product_inquiry: retrieves relevant knowledge-base chunks and answers via Groq.
    For greeting: responds with a warm welcome message (no RAG lookup needed).

    Updates:  messages
    """
    user_msg = _last_user_message(state)
    intent = state.get("intent", "product_inquiry")
    llm = _get_llm(temperature=0.5)

    if intent == "greeting":
        messages = [
            SystemMessage(content=GREETING_SYSTEM_PROMPT),
            LCHuman(content=user_msg),
        ]
    else:
        # Retrieve relevant passages from FAISS
        context = retrieve(user_msg, k=3)
        system = RAG_SYSTEM_PROMPT.format(context=context)
        messages = [
            SystemMessage(content=system),
            LCHuman(content=user_msg),
        ]

    response = llm.invoke(messages)
    reply = response.content.strip()

    return {"messages": _append_assistant(state, reply)}


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — lead_collection_node
# ─────────────────────────────────────────────────────────────────────────────

def lead_collection_node(state: dict) -> dict:
    """
    Collects lead fields ONE AT A TIME in this fixed order:
      name → email → platform

    On each call it checks which field is still missing and asks ONLY for
    the next missing field. It never asks for more than one field per turn.

    Updates:  messages + whichever single field was just provided by the user.
    """
    user_msg = _last_user_message(state)
    updates: dict = {}

    # If we just switched to the lead flow from another intent, ask for the name first.
    # We do NOT want to use their initial message (e.g. "I want to sign up") as their name.
    if not state.get("in_lead_flow"):
        updates["in_lead_flow"] = True
        reply = "Awesome! I'd love to help you get started. What's your name?"
        updates["messages"] = _append_assistant(state, reply)
        return updates

    name     = state.get("name")
    email    = state.get("email")
    platform = state.get("platform")

    # ── Parse the user's latest message into the next expected field ──────────
    # We fill fields strictly in order; whichever is first-missing gets filled.
    if not name:
        updates["name"] = user_msg.strip()
        reply = (
            f"Great to meet you, {updates['name']}! 😊\n"
            "What's your email address so we can send you the details?"
        )

    elif not email:
        updates["email"] = user_msg.strip()
        reply = (
            "Perfect! And which platform do you mainly publish on? "
            "(e.g. YouTube, TikTok, Instagram)"
        )

    elif not platform:
        updates["platform"] = user_msg.strip()
        reply = (
            f"Awesome — {updates.get('platform')} is a great choice! "
            "Let me get everything set up for you …"
        )
        updates["in_lead_flow"] = False

    else:
        # All fields already collected — shouldn't normally reach here
        reply = "Thanks! We already have your details. Is there anything else I can help with?"

    updates["messages"] = _append_assistant(state, reply)
    return updates


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — lead_capture_node
# ─────────────────────────────────────────────────────────────────────────────

def lead_capture_node(state: dict) -> dict:
    """
    Pre-fire guard: only calls mock_lead_capture() when ALL THREE fields
    (name, email, platform) are confirmed present in state.

    Updates:  messages, lead_captured
    """
    name     = state.get("name", "")
    email    = state.get("email", "")
    platform = state.get("platform", "")

    # Safety guard — this should never be False if routing is correct,
    # but we defend against it explicitly per spec.
    if not (name and email and platform):
        reply = "I still need a bit more information before I can complete your sign-up."
        return {"messages": _append_assistant(state, reply), "lead_captured": False}

    # Fire the tool
    mock_lead_capture(name, email, platform)

    reply = (
        f"🎉 You're all set, {name}! We've registered your interest in AutoStream.\n"
        f"A welcome email is on its way to {email}. "
        f"We'll make sure your {platform} content looks stunning. "
        "Feel free to ask if you have any more questions!"
    )

    return {
        "messages": _append_assistant(state, reply),
        "lead_captured": True,
    }
