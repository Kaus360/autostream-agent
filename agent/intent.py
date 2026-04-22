"""
agent/intent.py
Classifies every user message into exactly one of three intents using
the Groq LLM so the LangGraph router knows which node to activate next.

Intents:
  greeting        — casual hello / intro messages
  product_inquiry — questions about features, pricing, plans, policies
  high_intent     — user is ready to sign up / buy / try the product
"""

import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "llama3-70b-8192"

INTENT_SYSTEM_PROMPT = """You are an intent classifier for AutoStream, a SaaS video-editing platform.

Classify the user message into EXACTLY ONE of these three intents:
  - greeting        : The user is saying hello, introducing themselves, or making small talk.
  - product_inquiry : The user is asking about features, pricing, plans, refund policy, or how the product works.
  - high_intent     : The user is expressing interest in signing up, buying, trying a demo, or wants to get started.

Rules:
  1. Reply with ONLY the single intent label — no punctuation, no explanation.
  2. If in doubt between product_inquiry and high_intent, choose high_intent.
  3. Never output anything other than one of the three labels above.
"""


def classify_intent(user_message: str) -> str:
    """
    Sends *user_message* to the Groq LLM and returns one of:
    'greeting' | 'product_inquiry' | 'high_intent'

    Falls back to 'product_inquiry' if the model returns an unexpected value.
    """
    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,          # deterministic classification
        max_tokens=10,          # we only need one short label
    )

    messages = [
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip().lower()

    valid_intents = {"greeting", "product_inquiry", "high_intent"}
    if raw in valid_intents:
        return raw

    # Partial-match fallback
    for intent in valid_intents:
        if intent in raw:
            return intent

    # Hard fallback — treat unknowns as product inquiry
    return "product_inquiry"
