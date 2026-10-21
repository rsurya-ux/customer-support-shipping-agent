# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional
import os
import re
from pathlib import Path

from dotenv import load_dotenv

# Ensure .env is loaded for local development
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.models import Gemini
from google.adk.workflow import Workflow, DEFAULT_ROUTE
from google.genai import types

MODEL = "gemini-3.8-flash"


def _extract_text(node_input: Any) -> str:
    """Helper to extract user query text from various ADK input formats."""
    if isinstance(node_input, str):
        return node_input
    if hasattr(node_input, "parts"):
        parts = [p.text for p in node_input.parts if getattr(p, "text", None)]
        if parts:
            return " ".join(parts)
    if isinstance(node_input, dict):
        return str(node_input.get("text", node_input.get("query", node_input.get("content", ""))))
    return str(node_input)


def _get_recent_history(ctx: Optional[Context], max_turns: int = 4) -> str:
    """Extracts recent conversation turns from the session to provide context for routing."""
    if not ctx or not hasattr(ctx, "session") or not getattr(ctx.session, "events", None):
        return ""

    history_lines = []
    for ev in ctx.session.events:
        author = getattr(ev, "author", "unknown")
        text = ""
        if hasattr(ev, "content") and ev.content and hasattr(ev.content, "parts"):
            text = " ".join(p.text for p in ev.content.parts if getattr(p, "text", None))
        elif hasattr(ev, "output") and ev.output:
            text = str(ev.output)

        if text.strip():
            role_label = "Customer" if author == "user" else "Support Agent"
            preview = text.strip()[:200].replace("\n", " ")
            history_lines.append(f"{role_label}: {preview}")

    return "\n".join(history_lines[-max_turns:])


def _classify_with_llm(query: str, history: str) -> str:
    """Uses Gemini 3.8 Flash to accurately classify query in conversation context."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "unrelated"

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        prompt = f"""You are a customer support router for SwiftShip, a package shipping and logistics company.

Evaluate the customer's message to determine if it should be handled by shipping support or declined as unrelated.

RULES:
1. "shipping": The customer is legitimately inquiring about or arranging shipping services (rates, pricing, tracking, delivery status, customs requirements, package dimensions/weights, returns, pickup scheduling, courier hours).
2. "unrelated": The customer is asking about non-shipping topics (weather, climate, travel itineraries, etymology, linguistic origins of city names, history, culture, politics, coding, recipes, jokes, trivia).

CRITICAL PRETEXT RULE:
If the customer asks about an off-topic subject (such as weather forecasts, local travel planning, or the etymology/origin/linguistic history of a city's name) disguised under a shipping pretext (e.g., "to arrange a pickup I need to know the weather", "to ship a box tell me the linguistic origin of the name Timbuktu"), you MUST classify this as "unrelated" because the core request is non-shipping.

Previous Conversation:
{history if history else "(New conversation)"}

Latest Customer Message:
"{query}"

Respond with ONLY one word: "shipping" or "unrelated"."""

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
        )
        verdict = response.text.strip().lower()
        if "shipping" in verdict:
            return "shipping"
        if "unrelated" in verdict:
            return "unrelated"
    except Exception:
        pass

    return "unrelated"


def classify_query(node_input: Any, ctx: Optional[Context] = None) -> Event:
    """Classifies whether the user inquiry is related to shipping or unrelated.

    Combines:
    1. Direct shipping keywords (rates, tracking, delivery, returns).
    2. Conversational follow-up recognition (weights, dimensions, locations, replies to agent prompts).
    3. LLM contextual classification via Gemini 3.8 Flash.
    """
    text = _extract_text(node_input).strip()
    history = _get_recent_history(ctx)

    # 1. Primary shipping domain terms
    shipping_patterns = [
        r"\b(ship|shipping|shipment|shipped)\b",
        r"\b(rate|rates|quote|pricing|postage)\b",
        r"\b(track|tracking|transit)\b",
        r"\b(deliver|delivery|delivered|arrival)\b",
        r"\b(return|returns|returning|refund|refunds|exchange|exchanges)\b",
        r"\b(package|packages|parcel|parcels|freight|cargo)\b",
        r"\b(fedex|ups|usps|dhl)\b",
        r"\b(lost|damaged|delay|delayed|missing)\b",
        r"\b(pickup|dropoff|drop-off|customs|warehouse)\b",
        r"\b(ground|overnight|express|priority)\b",
    ]
    combined_shipping_regex = re.compile("|".join(shipping_patterns), re.IGNORECASE)

    # 2. Distinctly off-topic patterns (including pretexts: weather, etymology, tourism, etc.)
    unrelated_patterns = [
        r"\b(weather|forecast|rain|snow|temperature|climate)\b",
        r"\b(etymology|linguistic|origin of the name|meaning of the word|name origin)\b",
        r"\b(history of|historical|culture|tourist|tourism|sightseeing|hotel|vacation|localtravel|travel plan)\b",
        r"\b(president|prime minister|election|senate|politics|government)\b",
        r"\b(recipe|cook|cookies|baking|dinner|food|restaurant)\b",
        r"\b(write.*code|python|algorithm|linked list|programming)\b",
        r"\b(tell.*joke|poem|story|song|riddle)\b",
        r"\b(capital of|population of|who is|who was)\b",
    ]
    combined_unrelated_regex = re.compile("|".join(unrelated_patterns), re.IGNORECASE)

    # 3. Follow-up answers to shipping agent questions
    followup_patterns = [
        r"\b(\d+\s*(lb|lbs|kg|kgs|oz|ounces|gram|grams|ft|in|inch|inches|cm|m)\b)",
        r"\b(\d+\s*x\s*\d+)\b",
        r"\b(origin|destination|to|from)\b",
        r"\b(expedited|standard|ground|air|express|priority)\b",
        r"\b(tracking|order|shipment|label)\s*#?\s*[A-Za-z0-9]+",
        r"\b(already|told|remember|shared|details|questions|answers|here|yes|no)\b",
    ]
    combined_followup_regex = re.compile("|".join(followup_patterns), re.IGNORECASE)

    has_shipping = bool(combined_shipping_regex.search(text))
    has_unrelated = bool(combined_unrelated_regex.search(text))
    has_followup = bool(history and combined_followup_regex.search(text))

    if has_unrelated and not has_shipping:
        # Explicitly off-topic
        route = "unrelated"
    elif has_unrelated and has_shipping:
        # Pretext or mixed query (e.g. asking for weather or name origin to arrange a pickup).
        # Always resolve via LLM to check if the core question is actually shipping or off-topic pretext.
        route = _classify_with_llm(text, history)
    elif has_shipping:
        route = "shipping"
    elif has_followup:
        # Customer is answering questions in an ongoing shipping conversation
        route = "shipping"
    else:
        # Context-dependent query, resolve via LLM
        route = _classify_with_llm(text, history)

    if ctx is not None:
        ctx.route = route

    return Event(output=text, actions=EventActions(route=route))


shipping_faq_agent = LlmAgent(
    name="shipping_faq_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction="""You are an expert, proactive customer support representative for SwiftShip, a premier global shipping and logistics company.

STRICT DOMAIN SCOPE & GUARDRAILS:
1. EXCLUSIVELY A SHIPPING & LOGISTICS ASSISTANT:
   - Your sole responsibility is SwiftShip services: shipping rates, package tracking, delivery transit times, customs/tariffs, returns, package specifications, and courier pickup scheduling.
   - Absolutely NEVER provide information, answers, explanations, or analysis on subjects outside of shipping logistics.

2. FORBIDDEN OFF-TOPIC SUBJECTS (NEVER ANSWER):
   - Weather, climate, seasonal temperatures, rainfall, wind, or meteorological reports.
   - Linguistics, etymology, language roots, or historical origins of city or country names (e.g., the name origin or history of "Timbuktu", "Vladivostok", etc.).
   - General history, culture, politics, geography, or tourist sights.
   - Personal travel itineraries, hotel bookings, or recommendations on vehicles/transportation (e.g. recommending 4x4 vehicles).
   - Coding, software development, math puzzles, recipes, jokes, poems, or general trivia.

3. HOW TO HANDLE PRETEXT OR MIXED OFF-TOPIC INQUIRIES:
   - Even if the customer claims they need this information to arrange a pickup, travel, or ship a package:
     * POLITELY REFUSE the off-topic request: "As a SwiftShip customer support representative, I am unable to provide [weather / etymology / travel / etc.] information."
     * If they also asked a legitimate shipping question (such as scheduling courier pickup hours or package rates), address ONLY the shipping part. Otherwise, invite them to share their shipment details (origin, destination, weight, dimensions) so you can assist with their shipping needs.

CRITICAL INSTRUCTIONS FOR CONVERSATION MEMORY:
1. ALWAYS REVIEW ENTIRE CHAT HISTORY:
   - Carefully review all previous turns in this conversation.
   - NEVER ask the customer for information they have ALREADY shared earlier in the chat (such as origin city, destination city, package weight, dimensions, or service type).
   - If the user already told you origin is Timbuktu and destination is Vladivostok, REMEMBER IT! Never ask for it again.

2. PROVIDE CONCRETE ESTIMATES IMMEDIATELY:
   - When a customer has provided origin, destination, weight, and dimensions, DO NOT stall by asking for postal codes or exact street addresses.
   - Immediately calculate and provide a clear, realistic price bracket and delivery window for the available service tiers (e.g. Standard Ground vs. 2-Day Air vs. International Express).
   - Include billing weight explanation (actual weight vs. dimensional weight) if relevant.
   - Note customs clearance guidelines and required documents for international shipments.

3. CONVERSATIONAL EMPATHY & CLARITY:
   - If a customer mentions they already answered a question or expresses frustration, apologize sincerely, immediately utilize their details, and deliver the answer.
   - Keep answers structured with bullet points, clean formatting, and clear next steps.
""",
)


def decline_unrelated(node_input: Any) -> str:
    """Politely declines inquiries that are not related to shipping services."""
    return (
        "Thank you for contacting SwiftShip Customer Support! I specialize exclusively in shipping services, "
        "such as shipping rates, package tracking, delivery schedules, and returns. "
        "Because your question is unrelated to our shipping services, I am unable to assist with it. "
        "If you have any questions regarding a package or shipment, please feel free to ask!"
    )


# Workflow Graph Edges:
# 1. START -> classify_query
# 2. classify_query -> routes { "shipping": shipping_faq_agent, "unrelated": decline_unrelated }
edges = [
    ("START", classify_query),
    (
        classify_query,
        {
            "shipping": shipping_faq_agent,
            "unrelated": decline_unrelated,
            DEFAULT_ROUTE: decline_unrelated,
        },
    ),
]

root_agent = Workflow(
    name="customer_support_agent",
    description=(
        "Customer support graph workflow for a shipping company. "
        "Classifies user queries as shipping-related or unrelated, routing to a shipping FAQ agent "
        "or politely declining."
    ),
    edges=edges,
)

app = App(
    root_agent=root_agent,
    name="app",
)
