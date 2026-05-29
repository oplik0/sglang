# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from sglang.srt.function_call.core_types import StructureInfo, _GetInfoFunc
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector


class MiMoDetector(Qwen3CoderDetector):
    """
    Detector for MiMo function call format.

    Format:
        <tool_call>
        <function=execute_bash>
        <parameter=command>pwd && ls</parameter>
        </function>
        </tool_call>

    MiMo uses the same XML tool call format as Qwen3Coder.
    Reference: https://huggingface.co/XiaomiMiMo/MiMo-V2.5
    """

    def get_structural_tag_name(self) -> str:
        return "mimo"

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<tool_call>\n<function=' + name + '>',
            end='</function>\n</tool_call>',
            trigger='<tool_call>',
        )
