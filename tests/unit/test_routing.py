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

import pytest
from unittest.mock import MagicMock
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.events.event import Event
from google.genai import types

from app.agent import (
    classify_query,
    decline_unrelated,
    root_agent,
)


class TestCustomerSupportWorkflow:
    @pytest.mark.parametrize(
        "query, expected_route",
        [
            ("What are your shipping rates for express delivery?", "shipping"),
            ("Can you help me track my package?", "shipping"),
            ("How long does standard ground delivery take to arrive?", "shipping"),
            ("I need to return an item. How do I get a return label?", "shipping"),
            ("My parcel was damaged in transit, how can I file a claim?", "shipping"),
            ("What are the international customs fees for shipping to the UK?", "shipping"),
            ("What is the weather like in New York today?", "unrelated"),
            ("Who was the 16th president of the United States?", "unrelated"),
            ("Can you write python code to reverse a linked list?", "unrelated"),
            ("What is a good recipe for chocolate chip cookies?", "unrelated"),
            ("What is the linguistic origin of the name Timbuktu?", "unrelated"),
            ("Can you explain the etymology of the word Vladivostok for my shipment?", "unrelated"),
            ("Hey to arrange a pickup at timbuktu i need to plan localtravel there for which i need to know the weather", "unrelated"),
        ],
    )
    def test_classify_query_routing(self, query: str, expected_route: str) -> None:
        """Verify queries are accurately classified into shipping or unrelated routes."""
        event = classify_query(query)
        assert event.actions.route == expected_route
        assert event.output == query

        content_input = types.Content(
            role="user", parts=[types.Part.from_text(text=query)]
        )
        content_event = classify_query(content_input)
        assert content_event.actions.route == expected_route
        assert content_event.output == query

    def test_followup_answers_in_shipping_conversation(self) -> None:
        """Verify customer answers to agent questions (weights, dimensions, locations) stay routed to shipping."""
        # Simulate active session context where agent asked for package specs
        mock_ctx = MagicMock()
        mock_event_1 = Event(author="user", content=types.Content(role="user", parts=[types.Part.from_text(text="What is the shipping cost to Vladivostok?")]))
        mock_event_2 = Event(author="shipping_faq_agent", content=types.Content(role="model", parts=[types.Part.from_text(text="Please share the weight and dimensions.")]))
        mock_ctx.session.events = [mock_event_1, mock_event_2]

        # Customer responds with specs
        event = classify_query("30 lb, 2 ft x 2 ft x 3 ft, expedited, books", ctx=mock_ctx)
        assert event.actions.route == "shipping"

        # Customer responds with cities
        event_loc = classify_query("origin - timbuctoo ; destination - bladivostok", ctx=mock_ctx)
        assert event_loc.actions.route == "shipping"

    def test_decline_unrelated_message(self) -> None:
        """Verify the decline node produces a polite decline message explaining the shipping scope."""
        response = decline_unrelated("What is the speed of light?")
        assert isinstance(response, str)
        assert "Customer Support" in response
        assert "shipping" in response.lower()
        assert "unrelated" in response.lower()

    def test_workflow_graph_structure(self) -> None:
        """Verify workflow graph definition and root agent attributes."""
        assert root_agent.name == "customer_support_agent"
        assert len(root_agent.edges) == 2

    def test_end_to_end_unrelated_workflow_execution(self) -> None:
        """Verify the full workflow executes end-to-end and routes unrelated queries to the decline node."""
        session_service = InMemorySessionService()
        session = session_service.create_session_sync(user_id="test_user", app_name="test_app")
        runner = Runner(agent=root_agent, session_service=session_service, app_name="test_app")

        user_message = types.Content(
            role="user",
            parts=[types.Part.from_text(text="What is the capital of France?")]
        )

        events = list(
            runner.run(
                new_message=user_message,
                user_id="test_user",
                session_id=session.id,
            )
        )

        assert len(events) >= 1
        final_event = events[-1]
        assert "shipping services" in final_event.output.lower()
        assert "unrelated" in final_event.output.lower()
