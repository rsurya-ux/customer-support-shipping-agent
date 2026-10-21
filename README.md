# customer-support-agent

An AI customer support representative agent for a shipping company built using the **Google Agent Development Kit (ADK) 2.0 Graph Workflow API**.

---

## Architecture & Workflow Graph

The agent is implemented as an explicit directed workflow graph using `google.adk.workflow.Workflow`:

```
              ┌─────────────────────┐
              │        START        │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    classify_query   │
              └──────────┬──────────┘
                         │
         ┌───────────────┴───────────────┐
         │ (route="shipping")            │ (route="unrelated")
         ▼                               ▼
┌──────────────────┐           ┌────────────────────┐
│shipping_faq_agent│           │ decline_unrelated  │
│    (LlmAgent)    │           │  (FunctionNode)    │
└──────────────────┘           └────────────────────┘
```

### Nodes:
1. **`classify_query`**: Analyzes the customer's query to determine if it relates to shipping services (rates, tracking, delivery windows, returns, claims). Emits `route="shipping"` or `route="unrelated"`.
2. **`shipping_faq_agent` (`LlmAgent`)**: Powered by `gemini-3.8-flash`. Answers questions regarding shipping rates, tracking package status, delivery schedules, and returns/claims.
3. **`decline_unrelated` (Function Node)**: Politely explains that the agent specializes exclusively in shipping and logistics, declining non-shipping inquiries.

---

## Project Structure

```text
customer-support-agent/
├── app/
│   ├── __init__.py
│   ├── agent.py                 # Core ADK 2.0 Workflow graph definition
│   ├── fast_api_app.py          # FastAPI server
│   └── app_utils/
├── tests/
│   ├── unit/
│   │   ├── test_routing.py      # Unit & workflow integration tests
│   │   └── test_dummy.py
│   └── integration/
├── agents-cli-manifest.yaml     # Agents CLI project metadata
├── pyproject.toml               # Project dependencies (google-adk >= 2.12.0)
└── README.md
```

---

## Running the Agent

### 1. Run Unit Tests
```bash
.venv/bin/pytest tests/unit/test_routing.py -v
```

### 2. Run the Interactive Playground
```bash
agents-cli playground
```
This opens the local web playground in your browser where you can chat interactively with the customer support workflow.
