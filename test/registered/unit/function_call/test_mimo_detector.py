"""Unit tests for MiMoDetector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.mimo_detector import MiMoDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestMiMoDetector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Search the web",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                        },
                        "required": ["query"],
                    },
                ),
            ),
        ]
        self.detector = MiMoDetector()

    # ==================== has_tool_call Tests ====================

    def test_has_tool_call_true(self):
        self.assertTrue(self.detector.has_tool_call("<tool_call>"))

    def test_has_tool_call_false(self):
        self.assertFalse(self.detector.has_tool_call("Hello world"))

    # ==================== detect_and_parse Tests ====================

    def test_single_tool_call(self):
        text = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Beijing</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")

    def test_normal_text_before_tool_call(self):
        text = (
            "Let me check the weather.\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Beijing</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.normal_text, "Let me check the weather.\n")

    def test_no_tool_call(self):
        text = "The weather is nice today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "The weather is nice today.")

    def test_multiple_tool_calls(self):
        text = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Beijing</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "<function=search>\n"
            "<parameter=query>restaurants</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "search")

    def test_tool_call_with_multiple_arguments(self):
        text = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>London</parameter>\n"
            "<parameter=unit>celsius</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "London")
        self.assertEqual(args["unit"], "celsius")

    # ==================== Streaming Tests ====================

    def test_streaming_single_tool_call(self):
        detector = MiMoDetector()
        chunks = [
            "<tool_call>\n<function=get_weather>\n<parameter=city>Beijing",
            "</parameter>\n</function>\n</tool_call>",
        ]
        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "get_weather")

        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        params = json.loads(full_params)
        self.assertEqual(params["city"], "Beijing")

    def test_streaming_normal_text_before_tool(self):
        detector = MiMoDetector()
        result = detector.parse_streaming_increment(
            "Let me check the weather. ", self.tools
        )
        self.assertEqual(result.normal_text, "Let me check the weather. ")
        self.assertEqual(len(result.calls), 0)

    def test_streaming_text_then_tool_call(self):
        detector = MiMoDetector()
        chunks = [
            "I'll look that up. ",
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Tokyo</parameter>\n"
            "<parameter=unit>celsius</parameter>\n"
            "</function>\n"
            "</tool_call>",
        ]
        all_calls = []
        all_normal_text = ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)
            all_normal_text += result.normal_text

        self.assertEqual(all_normal_text, "I'll look that up. ")
        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "get_weather")
        full_params = "".join(c.parameters for c in all_calls if c.parameters)
        params = json.loads(full_params)
        self.assertEqual(params["city"], "Tokyo")
        self.assertEqual(params["unit"], "celsius")

    def test_streaming_trailing_text_after_tool_call(self):
        """Trailing text after </tool_call> should be preserved and emitted as normal text."""
        detector = MiMoDetector()
        chunks = [
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Beijing</parameter>\n"
            "</function>\n"
            "</tool_call> Hello world",
        ]
        all_normal_text = ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_normal_text += result.normal_text

        self.assertEqual(all_normal_text, " Hello world")

    def test_end_of_generation_tracking(self):
        detector = MiMoDetector()
        chunks = [
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>Beijing</parameter>\n"
            "</function>\n"
            "</tool_call>",
        ]
        for chunk in chunks:
            detector.parse_streaming_increment(chunk, self.tools)

        self.assertEqual(len(detector.prev_tool_call_arr), 1)
        self.assertEqual(detector.prev_tool_call_arr[0]["name"], "get_weather")
        self.assertEqual(
            detector.prev_tool_call_arr[0]["arguments"], {"city": "Beijing"}
        )
        self.assertEqual(
            detector.streamed_args_for_tool[0], '{"city": "Beijing"}'
        )

    # ==================== HTML entity unescaping Tests ====================

    def test_html_entity_unescaping(self):
        text = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>San &amp; Francisco</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "San & Francisco")

    # ==================== Unknown tool validation Tests ====================

    def test_unknown_tool_filtered(self):
        text = (
            "<tool_call>\n"
            "<function=unknown_tool>\n"
            "<parameter=city>Beijing</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        # Unknown tool should be filtered out (SGLANG_FORWARD_UNKNOWN_TOOLS defaults to False)
        self.assertEqual(len(result.calls), 0)

    # ==================== Type conversion Tests ====================

    def test_integer_type_conversion(self):
        """Verify that 'type': 'integer' (standard JSON Schema) is converted to int."""
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_days",
                    parameters={
                        "type": "object",
                        "properties": {
                            "days": {"type": "integer"},
                        },
                        "required": ["days"],
                    },
                ),
            ),
        ]
        text = (
            "<tool_call>\n"
            "<function=get_days>\n"
            "<parameter=days>5</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertIsInstance(args["days"], int)
        self.assertEqual(args["days"], 5)

    # ==================== structural tag Tests ====================

    def test_supports_structural_tag(self):
        self.assertTrue(self.detector.supports_structural_tag())

    def test_get_structural_tag_name(self):
        self.assertEqual(self.detector.get_structural_tag_name(), "mimo")

    def test_structure_info(self):
        info_func = self.detector.structure_info()
        info = info_func("get_weather")
        self.assertIn("get_weather", info.begin)
        self.assertIn("<tool_call>", info.trigger)
        self.assertIn("</function>", info.end)


if __name__ == "__main__":
    import unittest

    unittest.main()
