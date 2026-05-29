import json
import logging
import re
from typing import Any, Dict, List, Tuple

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)


class MinimaxM2Detector(BaseFormatDetector):
    """
    Detector for MiniMax M2 models.

    Assumes function call format:
        <minimax:tool_call>
        <invoke name="func1">
        <parameter name="param1">value1</parameter>
        <parameter name="param2">value2</parameter>
        </invoke>
        </minimax:tool_call>
    """

    def __init__(self):
        super().__init__()
        self.tool_call_start_token: str = "<minimax:tool_call>"
        self.tool_call_end_token: str = "</minimax:tool_call>"
        self.invoke_start_prefix: str = '<invoke name="'
        self.invoke_end_token: str = "</invoke>"
        self.parameter_start_prefix: str = '<parameter name="'
        self.parameter_end_token: str = "</parameter>"

        # Regex for non-streaming fallback
        self.tool_call_regex = re.compile(
            r"<minimax:tool_call>(.*?)</minimax:tool_call>|<minimax:tool_call>(.*?)$",
            re.DOTALL,
        )
        self.tool_call_function_regex = re.compile(
            r'<invoke name="(.*?)</invoke>|<invoke name="(.*)$', re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r'<parameter name="(.*?)</parameter>|<parameter name="(.*?)$', re.DOTALL
        )

        # Cursor-based streaming state
        self.parsed_pos: int = 0
        self.is_inside_tool_call: bool = False
        self.current_tool_param_count: int = 0
        self.json_started: bool = False
        self._current_function_name: str = ""
        self._accumulated_params: Dict[str, Any] = {}

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        normal, calls = self._extract(text, tools)
        return StreamingParseResult(normal_text=normal, calls=calls)

    def _convert_param_value(self, value: str, param_type: str) -> Any:
        """Convert parameter value to the correct type (legacy single-type version)."""
        return self._convert_param_value_with_types(value, [param_type])

    def _extract_types_from_schema(self, schema: Any) -> list[str]:
        """
        Extract all possible types from a JSON schema definition.
        Handles anyOf, oneOf, allOf, type arrays, and enum fields.
        """
        if schema is None:
            return ["string"]

        if not isinstance(schema, dict):
            return ["string"]

        types: set[str] = set()

        # Handle direct "type" field
        if "type" in schema:
            type_value = schema["type"]
            if isinstance(type_value, str):
                types.add(type_value)
            elif isinstance(type_value, list):
                for t in type_value:
                    if isinstance(t, str):
                        types.add(t)

        # Handle enum - infer types from enum values
        if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
            for value in schema["enum"]:
                if value is None:
                    types.add("null")
                elif isinstance(value, bool):
                    types.add("boolean")
                elif isinstance(value, int):
                    types.add("integer")
                elif isinstance(value, float):
                    types.add("number")
                elif isinstance(value, str):
                    types.add("string")
                elif isinstance(value, list):
                    types.add("array")
                elif isinstance(value, dict):
                    types.add("object")

        # Handle anyOf, oneOf, allOf - recursively extract types
        for choice_field in ("anyOf", "oneOf", "allOf"):
            if choice_field in schema and isinstance(schema[choice_field], list):
                for choice in schema[choice_field]:
                    extracted = self._extract_types_from_schema(choice)
                    types.update(extracted)

        # If no types found, default to string
        if not types:
            return ["string"]

        return list(types)

    def _convert_param_value_with_types(
        self, value: str, param_types: list[str]
    ) -> Any:
        """
        Convert parameter value to the correct type based on a list of possible types.
        Tries each type in order until one succeeds.
        """
        if value.lower() == "null":
            return None

        # Normalize types
        normalized_types = [t.lower() for t in param_types]

        # Try null first if it's in the list
        if "null" in normalized_types or value.lower() in ("null", "none", "nil"):
            return None

        # Try each type in order of preference (most specific first, string as fallback)
        type_priority = [
            "integer",
            "int",
            "number",
            "float",
            "boolean",
            "bool",
            "object",
            "array",
            "string",
            "str",
            "text",
        ]

        for param_type in type_priority:
            if param_type not in normalized_types:
                continue

            if param_type in ["string", "str", "text"]:
                return value
            elif param_type in ["integer", "int"]:
                try:
                    return int(value)
                except (ValueError, TypeError):
                    continue
            elif param_type in ["number", "float"]:
                try:
                    val = float(value)
                    return val if val != int(val) else int(val)
                except (ValueError, TypeError):
                    continue
            elif param_type in ["boolean", "bool"]:
                lower_val = value.lower().strip()
                if lower_val in ["true", "1", "yes", "on"]:
                    return True
                elif lower_val in ["false", "0", "no", "off"]:
                    return False
                continue
            elif param_type in ["object", "array"]:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    continue

        # Fallback: try JSON parse, then return as string
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def _get_param_types_from_config(
        self, param_name: str, param_config: dict
    ) -> list[str]:
        """
        Get parameter types from parameter configuration.
        """
        if param_name not in param_config:
            return ["string"]

        param_schema = param_config[param_name]
        if not isinstance(param_schema, dict):
            return ["string"]

        return self._extract_types_from_schema(param_schema)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Robust cursor-based streaming parser for MiniMax M2 tool calls.
        """
        self._buffer += new_text

        if not self._buffer:
            return StreamingParseResult()

        calls: List[ToolCallItem] = []
        normal_text_chunks: List[str] = []

        # Build tool indices for validation
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        while True:
            current_slice = self._buffer[self.parsed_pos :]

            if not current_slice:
                break

            # -------------------------------------------------------
            # 1. Priority detection: check if it's the start of Tool Call
            # -------------------------------------------------------
            if current_slice.startswith(self.tool_call_start_token):
                self.parsed_pos += len(self.tool_call_start_token)
                self.is_inside_tool_call = True
                self.current_tool_param_count = 0
                self.json_started = False
                self._current_function_name = ""
                self._accumulated_params = {}
                continue

            # -------------------------------------------------------
            # 2. Function Name: <invoke name="func_name">
            # -------------------------------------------------------
            if current_slice.startswith(self.invoke_start_prefix):
                end_quote = current_slice.find('">')
                if end_quote != -1:
                    func_name = current_slice[
                        len(self.invoke_start_prefix) : end_quote
                    ]

                    # Validate function name
                    if func_name not in self._tool_indices:
                        logger.warning(f"Unknown function: {func_name}")
                        if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                            # Return the unknown tool call block as normal text
                            self.parsed_pos += end_quote + 2
                            continue

                    self.current_tool_id += 1
                    self.current_tool_name_sent = True
                    self.current_tool_param_count = 0
                    self.json_started = False
                    self._current_function_name = func_name
                    self._accumulated_params = {}

                    # Ensure tracking arrays for end-of-generation checker
                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": func_name,
                        "arguments": {},
                    }
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")

                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )

                    self.parsed_pos += end_quote + 2
                    continue
                else:
                    # Incomplete tag
                    break

            # -------------------------------------------------------
            # 3. Parameter: <parameter name="param_name">value...
            # -------------------------------------------------------
            if current_slice.startswith(self.parameter_start_prefix):
                end_quote = current_slice.find('">')
                if end_quote != -1:
                    value_start_idx = end_quote + 2
                    rest_of_slice = current_slice[value_start_idx:]

                    # A parameter can end in multiple ways:
                    # 1. [Normal] Encounter </parameter>
                    # 2. [Abnormal] Encounter next <parameter name=
                    # 3. [Abnormal] Encounter </invoke>
                    cand_end_param = rest_of_slice.find(self.parameter_end_token)
                    cand_next_param = rest_of_slice.find(
                        self.parameter_start_prefix
                    )
                    cand_end_invoke = rest_of_slice.find(self.invoke_end_token)

                    candidates = []
                    if cand_end_param != -1:
                        candidates.append(
                            (cand_end_param, len(self.parameter_end_token))
                        )
                    if cand_next_param != -1:
                        candidates.append((cand_next_param, 0))
                    if cand_end_invoke != -1:
                        candidates.append((cand_end_invoke, 0))

                    if candidates:
                        best_cand = min(candidates, key=lambda x: x[0])
                        end_pos = best_cand[0]
                        end_token_len = best_cand[1]

                        param_name = current_slice[
                            len(self.parameter_start_prefix) : end_quote
                        ]
                        raw_value = rest_of_slice[:end_pos]

                        # Cleanup value (strip all leading/trailing newlines)
                        raw_value = raw_value.lstrip("\n").rstrip("\n")

                        # JSON Construction
                        if not self.json_started:
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    parameters="{",
                                )
                            )
                            self.json_started = True
                            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                                self.streamed_args_for_tool.append("")
                            self.streamed_args_for_tool[self.current_tool_id] += "{"

                        converted_val = self._parse_parameter(
                            self._current_function_name,
                            param_name,
                            raw_value,
                            tools,
                        )

                        json_key_val = f'{json.dumps(param_name)}: {json.dumps(converted_val, ensure_ascii=False)}'

                        if self.current_tool_param_count > 0:
                            fragment = f", {json_key_val}"
                        else:
                            fragment = json_key_val

                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                parameters=fragment,
                            )
                        )
                        self.current_tool_param_count += 1

                        # Update tracking arrays for end-of-generation checker
                        self._accumulated_params[param_name] = converted_val
                        self.prev_tool_call_arr[self.current_tool_id][
                            "arguments"
                        ] = self._accumulated_params.copy()
                        while len(self.streamed_args_for_tool) <= self.current_tool_id:
                            self.streamed_args_for_tool.append("")
                        self.streamed_args_for_tool[self.current_tool_id] += fragment

                        # Advance cursor
                        total_len = value_start_idx + end_pos + end_token_len
                        self.parsed_pos += total_len
                        continue

                # Incomplete parameter tag or value
                break

            # -------------------------------------------------------
            # 4. Function End: </invoke>
            # -------------------------------------------------------
            if current_slice.startswith(self.invoke_end_token):
                if not self.json_started:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            parameters="{",
                        )
                    )
                    self.json_started = True
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")
                    self.streamed_args_for_tool[self.current_tool_id] += "{"

                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        parameters="}",
                    )
                )
                self.parsed_pos += len(self.invoke_end_token)
                self._current_function_name = ""

                # Update tracking arrays for end-of-generation checker
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")
                self.streamed_args_for_tool[self.current_tool_id] += "}"
                self.prev_tool_call_arr[self.current_tool_id][
                    "arguments"
                ] = self._accumulated_params.copy()
                continue

            # -------------------------------------------------------
            # 5. Tool Call End: </minimax:tool_call>
            # -------------------------------------------------------
            if current_slice.startswith(self.tool_call_end_token):
                self.parsed_pos += len(self.tool_call_end_token)
                self.is_inside_tool_call = False
                continue

            # -------------------------------------------------------
            # 6. Handling content / whitespace / normal text
            # -------------------------------------------------------
            if not self.is_inside_tool_call:
                # Outside tool call: stream text until next start token
                next_start = current_slice.find(self.tool_call_start_token)
                if next_start == -1:
                    # Check for partial start token at the end
                    partial_len = self._ends_with_partial_token(
                        current_slice, self.tool_call_start_token
                    )
                    if partial_len > 0:
                        text_to_append = current_slice[:-partial_len]
                        if text_to_append:
                            normal_text_chunks.append(text_to_append)
                        self.parsed_pos += len(text_to_append)
                        break
                    else:
                        normal_text_chunks.append(current_slice)
                        self.parsed_pos += len(current_slice)
                        continue
                elif next_start == 0:
                    # Should have been handled above
                    continue
                else:
                    normal_text_chunks.append(current_slice[:next_start])
                    self.parsed_pos += next_start
                    continue
            else:
                # Inside tool call: discard whitespace/text between tags
                next_open = current_slice.find("<")

                if next_open == -1:
                    # Entire segment is text/whitespace inside tool call, discard
                    self.parsed_pos += len(current_slice)
                    continue
                elif next_open == 0:
                    # Starts with '<' but doesn't match any known tag
                    # Only include opening/start tags here; closing tags
                    # (like </parameter>) that appear without a matching
                    # opening tag should be skipped, not waited for.
                    possible_tags = [
                        self.tool_call_start_token,
                        self.tool_call_end_token,
                        self.invoke_start_prefix,
                        self.invoke_end_token,
                        self.parameter_start_prefix,
                    ]

                    is_potential_tag = False
                    for tag in possible_tags:
                        if tag.startswith(current_slice):
                            is_potential_tag = True
                            break

                    if is_potential_tag:
                        break  # Wait for more
                    else:
                        # Plain '<' symbol or unexpected closing tag
                        self.parsed_pos += 1
                        continue
                else:
                    # '<' is in the middle, skip whitespace before it
                    self.parsed_pos += next_open
                    continue

        # Memory Cleanup: Slice the buffer
        if self.parsed_pos > 0:
            self._buffer = self._buffer[self.parsed_pos :]
            self.parsed_pos = 0

        normal_text = "".join(normal_text_chunks) if normal_text_chunks else ""
        return StreamingParseResult(calls=calls, normal_text=normal_text)

    def _extract(self, text: str, tools: List[Tool]) -> Tuple[str, List[ToolCallItem]]:
        normal_parts: List[str] = []
        calls: List[ToolCallItem] = []
        cursor = 0
        while True:
            s = text.find(self.tool_call_start_token, cursor)
            if s == -1:
                normal_parts.append(text[cursor:])
                break
            normal_parts.append(text[cursor:s])
            e = text.find(self.tool_call_end_token, s)
            if e == -1:
                normal_parts.append(text[s:])
                break
            block = text[s : e + len(self.tool_call_end_token)]
            cursor = e + len(self.tool_call_end_token)
            calls.extend(self._parse_block(block, tools))
        return "".join(normal_parts), calls

    def _parse_block(self, block: str, tools: List[Tool]) -> List[ToolCallItem]:
        res: List[ToolCallItem] = []
        for m in self.tool_call_function_regex.findall(block):
            txt = m[0] if m[0] else m[1]
            if '">' not in txt:
                continue
            idx = txt.index('">')
            fname = txt[:idx].strip()
            body = txt[idx + 2 :]
            params: Dict[str, Any] = {}
            for pm in self.tool_call_parameter_regex.findall(body):
                ptxt = pm[0] if pm[0] else pm[1]
                if '">' not in ptxt:
                    continue
                pidx = ptxt.index('">')
                pname = ptxt[:pidx].strip()
                pval = ptxt[pidx + 2 :].lstrip("\n").rstrip("\n")
                params[pname] = self._parse_parameter(fname, pname, pval, tools)
            raw = {"name": fname, "arguments": params}
            try:
                res.extend(self.parse_base_json(raw, tools))
            except Exception:
                logger.warning("invalid tool call for %s dropped", fname)
        return res

    def _parse_parameter(
        self, fname: str, pname: str, pval: str, tools: List[Tool]
    ) -> Any:
        param_config = {}
        for tool in tools:
            if tool.function.name == fname and tool.function.parameters is not None:
                parameters = tool.function.parameters
                if isinstance(parameters, dict) and "properties" in parameters:
                    param_config = parameters["properties"]
                    break

        param_type = self._get_param_types_from_config(pname, param_config)
        return self._convert_param_value_with_types(pval, param_type)

    def supports_structural_tag(self) -> bool:
        return True

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<minimax:tool_call>\n<invoke name="' + name + '">',
            end="</invoke>\n</minimax:tool_call>",
            trigger="<minimax:tool_call>",
        )

    def get_structural_tag_name(self) -> str:
        return "minimax"
