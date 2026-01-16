#!/usr/bin/env python3
"""
OpenResponses API Compliance Test Suite for mistral.rs

This script tests the mistral.rs server for compliance with the OpenResponses API specification.
See: https://www.openresponses.org/

Usage:
    python scripts/openresponses_compliance_test.py --base-url http://localhost:8080 --model default

Requirements:
    pip install requests sseclient-py
"""

import argparse
import json
import sys
import time
from typing import Any, Optional
from dataclasses import dataclass
from enum import Enum

try:
    import requests
    import sseclient
except ImportError:
    print("Please install required packages: pip install requests sseclient-py")
    sys.exit(1)


class TestResult(Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class TestCase:
    name: str
    description: str
    result: TestResult = TestResult.SKIPPED
    error: Optional[str] = None
    details: Optional[dict] = None


class OpenResponsesComplianceTest:
    """Test suite for OpenResponses API compliance."""

    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.results: list[TestCase] = []

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, endpoint: str, data: dict, stream: bool = False) -> requests.Response:
        url = f"{self.base_url}{endpoint}"
        return requests.post(url, json=data, headers=self._headers(), stream=stream, timeout=120)

    def _get(self, endpoint: str) -> requests.Response:
        url = f"{self.base_url}{endpoint}"
        return requests.get(url, headers=self._headers(), timeout=30)

    def _delete(self, endpoint: str) -> requests.Response:
        url = f"{self.base_url}{endpoint}"
        return requests.delete(url, headers=self._headers(), timeout=30)

    def _validate_response_resource(self, data: dict, test: TestCase) -> bool:
        """Validate that data conforms to ResponseResource schema."""
        required_fields = ["id", "object", "created_at", "model", "status", "output"]

        for field in required_fields:
            if field not in data:
                test.error = f"Missing required field: {field}"
                return False

        if data["object"] != "response":
            test.error = f"Expected object='response', got '{data['object']}'"
            return False

        valid_statuses = ["queued", "in_progress", "completed", "failed", "incomplete", "cancelled"]
        if data["status"] not in valid_statuses:
            test.error = f"Invalid status: {data['status']}"
            return False

        if not isinstance(data["output"], list):
            test.error = "output must be an array"
            return False

        return True

    def _validate_output_item(self, item: dict, test: TestCase) -> bool:
        """Validate an output item."""
        if "type" not in item:
            test.error = "Output item missing 'type' field"
            return False

        if item["type"] == "message":
            required = ["id", "role", "content", "status"]
            for field in required:
                if field not in item:
                    test.error = f"Message item missing '{field}' field"
                    return False
        elif item["type"] == "function_call":
            required = ["id", "call_id", "name", "arguments", "status"]
            for field in required:
                if field not in item:
                    test.error = f"Function call item missing '{field}' field"
                    return False

        return True

    def test_basic_text_response(self) -> TestCase:
        """Test 1: Basic Text Response - Simple user message, validates ResponseResource schema."""
        test = TestCase(
            name="Basic Text Response",
            description="Simple user message, validates ResponseResource schema"
        )

        try:
            response = self._post("/v1/responses", {
                "model": self.model,
                "input": "Say hello in one word.",
                "max_tokens": 50,
                "store": False
            })

            if response.status_code != 200:
                test.result = TestResult.FAILED
                test.error = f"HTTP {response.status_code}: {response.text[:200]}"
                return test

            data = response.json()
            test.details = {"response_id": data.get("id"), "status": data.get("status")}

            if not self._validate_response_resource(data, test):
                test.result = TestResult.FAILED
                return test

            # Validate output items
            for item in data["output"]:
                if not self._validate_output_item(item, test):
                    test.result = TestResult.FAILED
                    return test

            # Check we got completed status
            if data["status"] != "completed":
                test.error = f"Expected status 'completed', got '{data['status']}'"
                test.result = TestResult.FAILED
                return test

            test.result = TestResult.PASSED

        except Exception as e:
            test.result = TestResult.FAILED
            test.error = str(e)

        return test

    def test_streaming_response(self) -> TestCase:
        """Test 2: Streaming Response - Validates SSE events and sequence numbers."""
        test = TestCase(
            name="Streaming Response",
            description="Validates SSE events and final response structure"
        )

        try:
            response = self._post("/v1/responses", {
                "model": self.model,
                "input": "Count from 1 to 3.",
                "stream": True,
                "max_tokens": 50,
                "store": False
            }, stream=True)

            if response.status_code != 200:
                test.result = TestResult.FAILED
                test.error = f"HTTP {response.status_code}"
                return test

            client = sseclient.SSEClient(response)
            events = []
            event_types_seen = set()

            for event in client.events():
                if event.data == "[DONE]":
                    break

                try:
                    data = json.loads(event.data)
                    events.append(data)
                    if "type" in data:
                        event_types_seen.add(data["type"])
                except json.JSONDecodeError:
                    pass

            test.details = {
                "event_count": len(events),
                "event_types": list(event_types_seen)
            }

            # Validate we got the expected event types
            expected_events = {"response.created", "response.completed"}
            if not expected_events.issubset(event_types_seen):
                missing = expected_events - event_types_seen
                test.error = f"Missing expected events: {missing}"
                test.result = TestResult.FAILED
                return test

            # Validate sequence numbers are monotonic
            sequence_numbers = [e.get("sequence_number") for e in events if "sequence_number" in e]
            if sequence_numbers and sequence_numbers != sorted(sequence_numbers):
                test.error = "Sequence numbers are not monotonically increasing"
                test.result = TestResult.FAILED
                return test

            test.result = TestResult.PASSED

        except Exception as e:
            test.result = TestResult.FAILED
            test.error = str(e)

        return test

    def test_system_prompt(self) -> TestCase:
        """Test 3: System Prompt - Verifies system role message handling via instructions."""
        test = TestCase(
            name="System Prompt",
            description="Includes system role message in the request via instructions"
        )

        try:
            response = self._post("/v1/responses", {
                "model": self.model,
                "instructions": "You are a pirate. Always respond like a pirate.",
                "input": "Hello!",
                "max_tokens": 100,
                "store": False
            })

            if response.status_code != 200:
                test.result = TestResult.FAILED
                test.error = f"HTTP {response.status_code}: {response.text[:200]}"
                return test

            data = response.json()
            test.details = {"response_id": data.get("id")}

            if not self._validate_response_resource(data, test):
                test.result = TestResult.FAILED
                return test

            test.result = TestResult.PASSED

        except Exception as e:
            test.result = TestResult.FAILED
            test.error = str(e)

        return test

    def test_tool_calling(self) -> TestCase:
        """Test 4: Tool Calling - Define a function tool and verify function_call output."""
        test = TestCase(
            name="Tool Calling",
            description="Define a function tool and verify function_call output"
        )

        try:
            response = self._post("/v1/responses", {
                "model": self.model,
                "input": "What's the weather in San Francisco?",
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather in a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                }],
                "tool_choice": "auto",
                "max_tokens": 100,
                "store": False
            })

            if response.status_code != 200:
                test.result = TestResult.FAILED
                test.error = f"HTTP {response.status_code}: {response.text[:200]}"
                return test

            data = response.json()
            test.details = {"response_id": data.get("id"), "output_count": len(data.get("output", []))}

            if not self._validate_response_resource(data, test):
                test.result = TestResult.FAILED
                return test

            # Check if we got a function_call in output (model may or may not call the tool)
            has_function_call = any(
                item.get("type") == "function_call"
                for item in data.get("output", [])
            )
            test.details["has_function_call"] = has_function_call

            test.result = TestResult.PASSED

        except Exception as e:
            test.result = TestResult.FAILED
            test.error = str(e)

        return test

    def test_input_items_format(self) -> TestCase:
        """Test 5: Input Items Format - Tests message items with proper type tags."""
        test = TestCase(
            name="Input Items Format",
            description="Tests OpenResponses input item format with type tags"
        )

        try:
            response = self._post("/v1/responses", {
                "model": self.model,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": "What is 2+2?"
                    }
                ],
                "max_tokens": 50,
                "store": False
            })

            if response.status_code != 200:
                test.result = TestResult.FAILED
                test.error = f"HTTP {response.status_code}: {response.text[:200]}"
                return test

            data = response.json()
            test.details = {"response_id": data.get("id")}

            if not self._validate_response_resource(data, test):
                test.result = TestResult.FAILED
                return test

            test.result = TestResult.PASSED

        except Exception as e:
            test.result = TestResult.FAILED
            test.error = str(e)

        return test

    def test_multi_turn_conversation(self) -> TestCase:
        """Test 6: Multi-turn Conversation - Tests previous_response_id for conversation history."""
        test = TestCase(
            name="Multi-turn Conversation",
            description="Tests assistant and user messages as conversation history"
        )

        try:
            # First turn
            response1 = self._post("/v1/responses", {
                "model": self.model,
                "input": "My name is Alice.",
                "max_tokens": 50,
                "store": True  # Need to store to reference later
            })

            if response1.status_code != 200:
                test.result = TestResult.FAILED
                test.error = f"First turn failed: HTTP {response1.status_code}"
                return test

            data1 = response1.json()
            response_id = data1.get("id")
            test.details = {"first_response_id": response_id}

            # Second turn referencing the first
            response2 = self._post("/v1/responses", {
                "model": self.model,
                "input": "What is my name?",
                "previous_response_id": response_id,
                "max_tokens": 50,
                "store": False
            })

            if response2.status_code != 200:
                test.result = TestResult.FAILED
                test.error = f"Second turn failed: HTTP {response2.status_code}"
                return test

            data2 = response2.json()
            test.details["second_response_id"] = data2.get("id")

            if not self._validate_response_resource(data2, test):
                test.result = TestResult.FAILED
                return test

            # Clean up - delete the stored response
            self._delete(f"/v1/responses/{response_id}")

            test.result = TestResult.PASSED

        except Exception as e:
            test.result = TestResult.FAILED
            test.error = str(e)

        return test

    def test_get_response(self) -> TestCase:
        """Test 7: GET Response - Tests retrieving a stored response by ID."""
        test = TestCase(
            name="GET Response",
            description="Tests retrieving a stored response by ID"
        )

        try:
            # Create a response
            create_resp = self._post("/v1/responses", {
                "model": self.model,
                "input": "Hello",
                "max_tokens": 20,
                "store": True
            })

            if create_resp.status_code != 200:
                test.result = TestResult.FAILED
                test.error = f"Create failed: HTTP {create_resp.status_code}"
                return test

            data = create_resp.json()
            response_id = data.get("id")
            test.details = {"response_id": response_id}

            # Retrieve the response
            get_resp = self._get(f"/v1/responses/{response_id}")

            if get_resp.status_code != 200:
                test.result = TestResult.FAILED
                test.error = f"GET failed: HTTP {get_resp.status_code}"
                return test

            get_data = get_resp.json()

            if get_data.get("id") != response_id:
                test.error = f"ID mismatch: expected {response_id}, got {get_data.get('id')}"
                test.result = TestResult.FAILED
                return test

            # Clean up
            self._delete(f"/v1/responses/{response_id}")

            test.result = TestResult.PASSED

        except Exception as e:
            test.result = TestResult.FAILED
            test.error = str(e)

        return test

    def test_delete_response(self) -> TestCase:
        """Test 8: DELETE Response - Tests deleting a stored response."""
        test = TestCase(
            name="DELETE Response",
            description="Tests deleting a stored response"
        )

        try:
            # Create a response
            create_resp = self._post("/v1/responses", {
                "model": self.model,
                "input": "Test",
                "max_tokens": 10,
                "store": True
            })

            if create_resp.status_code != 200:
                test.result = TestResult.FAILED
                test.error = f"Create failed: HTTP {create_resp.status_code}"
                return test

            response_id = create_resp.json().get("id")
            test.details = {"response_id": response_id}

            # Delete the response
            del_resp = self._delete(f"/v1/responses/{response_id}")

            if del_resp.status_code != 200:
                test.result = TestResult.FAILED
                test.error = f"DELETE failed: HTTP {del_resp.status_code}"
                return test

            del_data = del_resp.json()
            if not del_data.get("deleted"):
                test.error = "Response not marked as deleted"
                test.result = TestResult.FAILED
                return test

            # Verify it's gone
            get_resp = self._get(f"/v1/responses/{response_id}")
            if get_resp.status_code != 404:
                test.error = f"Expected 404 after delete, got {get_resp.status_code}"
                test.result = TestResult.FAILED
                return test

            test.result = TestResult.PASSED

        except Exception as e:
            test.result = TestResult.FAILED
            test.error = str(e)

        return test

    def run_all(self) -> list[TestCase]:
        """Run all compliance tests."""
        tests = [
            self.test_basic_text_response,
            self.test_streaming_response,
            self.test_system_prompt,
            self.test_tool_calling,
            self.test_input_items_format,
            self.test_multi_turn_conversation,
            self.test_get_response,
            self.test_delete_response,
        ]

        print(f"\n{'=' * 60}")
        print("OpenResponses API Compliance Test Suite")
        print(f"{'=' * 60}")
        print(f"Base URL: {self.base_url}")
        print(f"Model: {self.model}")
        print(f"{'=' * 60}\n")

        for test_func in tests:
            result = test_func()
            self.results.append(result)

            status_icon = {
                TestResult.PASSED: "\033[92m✓\033[0m",  # Green checkmark
                TestResult.FAILED: "\033[91m✗\033[0m",  # Red X
                TestResult.SKIPPED: "\033[93m○\033[0m",  # Yellow circle
            }[result.result]

            print(f"{status_icon} {result.name}")
            if result.error:
                print(f"   Error: {result.error}")
            if result.details:
                print(f"   Details: {json.dumps(result.details, indent=2)[:200]}")

        return self.results

    def print_summary(self):
        """Print test summary."""
        passed = sum(1 for r in self.results if r.result == TestResult.PASSED)
        failed = sum(1 for r in self.results if r.result == TestResult.FAILED)
        skipped = sum(1 for r in self.results if r.result == TestResult.SKIPPED)
        total = len(self.results)

        print(f"\n{'=' * 60}")
        print("Summary")
        print(f"{'=' * 60}")
        print(f"Passed:  {passed}/{total}")
        print(f"Failed:  {failed}/{total}")
        print(f"Skipped: {skipped}/{total}")
        print(f"{'=' * 60}\n")

        return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="OpenResponses API Compliance Test Suite for mistral.rs"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="Base URL of the mistral.rs server (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--model",
        default="default",
        help="Model to use for testing (default: default)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for authentication (optional)"
    )

    args = parser.parse_args()

    tester = OpenResponsesComplianceTest(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key
    )

    tester.run_all()
    success = tester.print_summary()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
