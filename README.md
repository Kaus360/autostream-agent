# AutoStream AI Agent

A conversational AI sales & support agent for **AutoStream** — an AI-powered automated video editing SaaS by **ServiceHive**. Built with LangChain, LangGraph, Groq LLM, and a local FAISS vector store.

---

## Features

- 🧠 **Intent Classification** — every message is classified into `greeting`, `product_inquiry`, or `high_intent` using Groq's `llama-3.3-70b-versatile`
- 📚 **RAG Pipeline** — product questions are answered from a local `knowledge_base.md` embedded with HuggingFace `all-MiniLM-L6-v2` and stored in FAISS (no paid embedding API)
- 🔁 **Multi-turn State** — full conversation history and lead fields persist across turns via LangGraph's `AgentState`
- 🎯 **Lead Collection** — collects `name → email → platform` one field at a time before firing the capture tool

---

## How to Run Locally

### 1. Clone & enter the project

```bash
git clone <your-repo-url>
cd autostream-agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API key

```bash
cp .env.example .env
# Open .env and paste your Groq API key
```

Get a free Groq API key at <https://console.groq.com>.

### 5. Run the agent

```bash
python main.py
```

---

## Architecture (~200 words)

The agent is built around **LangGraph**, a graph-based orchestration library that extends LangChain. LangGraph was chosen over a plain LangChain chain because it natively supports **stateful, multi-turn conversations** — each invocation of the graph receives the full `AgentState` and returns an updated version, making cross-turn memory trivial without external databases.

**State Management** is handled by a typed `AgentState` `TypedDict` containing: `messages` (full history), `intent` (last classified intent), `name`, `email`, `platform` (lead collection fields), and `lead_captured` (a guard flag). This state is passed through the graph on every turn and mutated only by the node that is responsible for a given field.

**Graph flow:**
1. `classify_intent` — Groq LLM classifies every message; a conditional edge routes to the correct next node.
2. `rag_response` — For greetings and product questions, FAISS retrieves the top-3 relevant knowledge-base chunks which are injected into a Groq prompt.
3. `lead_collection` — For high-intent users, fields are gathered one per turn (name → email → platform) using simple sequential logic.
4. `lead_capture_tool` — Only fires `mock_lead_capture()` once all three fields are confirmed — enforced by both the conditional routing edge and an in-node guard.

---

## WhatsApp Deployment via Twilio

To deploy this agent on WhatsApp, use **Twilio's WhatsApp API** with a public webhook endpoint.

### Overview

```
WhatsApp User
    │  (sends message)
    ▼
Twilio WhatsApp Sandbox / Business Number
    │  (HTTP POST to your webhook)
    ▼
Your Web Server  ← FastAPI or Flask
    │  (extracts message, invokes LangGraph agent)
    ▼
AutoStream Agent (this project)
    │  (returns reply)
    ▼
Twilio API  →  WhatsApp User
```

### Step-by-step

#### 1. Install a web framework

```bash
pip install fastapi uvicorn twilio
```

#### 2. Create a webhook endpoint (`whatsapp_webhook.py`)

```python
from fastapi import FastAPI, Form
from twilio.twiml.messaging_response import MessagingResponse
from agent.graph import build_graph, AgentState

app = FastAPI()
graph = build_graph()

# In production store sessions in Redis / a DB; dict is fine for local testing
sessions: dict[str, AgentState] = {}

@app.post("/whatsapp")
async def whatsapp_webhook(From: str = Form(...), Body: str = Form(...)):
    sender = From  # e.g. "whatsapp:+14155238886"

    # Retrieve or initialise session state for this sender
    state = sessions.get(sender, {
        "messages": [], "intent": "", "name": None,
        "email": None, "platform": None, "lead_captured": False,
    })

    state["messages"].append({"role": "user", "content": Body})
    state = graph.invoke(state)
    sessions[sender] = state

    # Extract latest assistant reply
    reply = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "assistant"),
        "Sorry, I couldn't process that."
    )

    twiml = MessagingResponse()
    twiml.message(reply)
    return str(twiml)
```

#### 3. Expose your server publicly

```bash
uvicorn whatsapp_webhook:app --port 8000

# In a second terminal, use ngrok to get a public URL:
ngrok http 8000
```

#### 4. Configure Twilio

1. Go to **Twilio Console → Messaging → Try it out → Send a WhatsApp message**.
2. In the **Sandbox Settings**, set the **"When a message comes in"** webhook URL to:
   ```
   https://<your-ngrok-id>.ngrok.io/whatsapp
   ```
   with method `HTTP POST`.
3. Send a message to your Twilio sandbox number on WhatsApp — the agent will respond.

#### 5. Production considerations

| Concern | Recommendation |
|---|---|
| Session persistence | Store `AgentState` per sender in **Redis** or **PostgreSQL** |
| Scaling | Deploy with **Gunicorn + Uvicorn workers** on Railway, Render, or AWS |
| WhatsApp Business | Apply for a Twilio WhatsApp Business number for production use |
| Security | Validate the `X-Twilio-Signature` header on every incoming request |

---

## Project Structure

```
autostream-agent/
├── main.py              # CLI entry point
├── agent/
│   ├── graph.py         # LangGraph StateGraph + AgentState
│   ├── nodes.py         # All node functions
│   ├── intent.py        # Groq-based intent classifier
│   └── tools.py         # mock_lead_capture tool
├── rag/
│   ├── loader.py        # Markdown loader + text splitter
│   └── retriever.py     # FAISS vector store + retrieve()
├── knowledge_base.md    # Product knowledge source
├── requirements.txt
├── .env.example
└── README.md
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq Cloud API key (required) |

---

## License

MIT © ServiceHive
