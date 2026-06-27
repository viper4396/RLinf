# Copyright 2025 The RLinf Authors.
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


def get_tools_description(
    max_workers_per_planner: int,
    max_toolcall_per_worker: int,
) -> dict:
    """Build the English tool descriptions with configurable call limits.

    The per-call limits are injected from config instead of being hardcoded, so
    that changing ``max_workers_per_planner`` / ``max_toolcall_per_worker`` keeps
    the tool descriptions consistent with the parser's enforcement.

    Args:
        max_workers_per_planner: Maximum number of sub-agents a planner may create
            in a single ``create_sub_agents`` call.
        max_toolcall_per_worker: Maximum number of ``search`` / ``access`` tool
            instances a worker may issue in a single call.

    Returns:
        Mapping from tool key to its OpenAI-style tool schema.
    """
    return {
        "create_sub_agents": {
            "type": "function",
            "function": {
                "name": "create_sub_agents",
                "description": (
                    "Creates sub-agents that can perform specific tasks based on "
                    "the input prompt. You can create multiple sub-agents "
                    "concurrently within a single call, but you are limited to "
                    f"creating a maximum of {max_workers_per_planner} sub-agents "
                    "in any given call."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sub_agents": {
                            "type": "array",
                            "description": "The sub-agents to create. Each sub-agent is created and executed in parallel; there is no order or sequence among them.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "prompt": {
                                        "type": "string",
                                        "description": "The specific details of the subtask that the sub-agent needs to complete.",
                                    }
                                },
                                "required": ["prompt"],
                            },
                        },
                    },
                    "required": ["sub_agents"],
                },
            },
        },
        "access": {
            "type": "function",
            "function": {
                "name": "access",
                "description": (
                    "This is a link-reading tool that opens webpages and retrieves "
                    "information from them based on your intent. You may access "
                    "multiple URLs simultaneously in a single call, but you are "
                    f"limited to a maximum of {max_toolcall_per_worker} tool "
                    "instances per call."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "urls": {
                            "type": "array",
                            "description": "The list of URLs to access. Each access tool is created and executed in parallel; there is no order or sequence among them.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "url": {
                                        "type": "string",
                                        "description": "Target link: should be a complete URL. Remember to only use the URLs provided by the search tool",
                                    },
                                    "info_to_extract": {
                                        "type": "string",
                                        "description": "The specific question or information to extract from this URL",
                                    },
                                },
                                "required": ["url", "info_to_extract"],
                            },
                        },
                    },
                    "required": ["urls"],
                },
            },
        },
        "search": {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    "This is a search tool. Enter search queries, and it will "
                    "return a list of web pages along with their corresponding "
                    "summary information. You may search multiple queries "
                    "simultaneously in a single call, but you are limited to a "
                    f"maximum of {max_toolcall_per_worker} tool instances per call."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "description": "The list of search queries. Each search tool is created and executed in parallel; there is no order or sequence among them.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "question to be searched.",
                                    },
                                    "count": {
                                        "type": "integer",
                                        "description": "The number of results to return. Must be less than 10, and default is 3",
                                        "default": 3,
                                    },
                                },
                                "required": ["query"],
                            },
                        },
                    },
                    "required": ["queries"],
                },
            },
        },
        "access_single_agent": {
            "type": "function",
            "function": {
                "name": "access",
                "description": "This is a link-reading tool that opens webpages and retrieves information from them based on your intent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Target link: should be a complete URL. Remember to only use the URLs provided by the search tool",
                        },
                        "info_to_extract": {
                            "type": "string",
                            "description": "The specific question or information to extract from the URL",
                        },
                    },
                    "required": ["url", "info_to_extract"],
                },
            },
        },
        "search_single_agent": {
            "type": "function",
            "function": {
                "name": "search",
                "description": "This is a search tool. Enter search queries, and it will return a list of web pages along with their corresponding summary information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Question to be searched.",
                        },
                        "count": {
                            "type": "integer",
                            "description": "The number of results to return. Must be less than 10, and default is 3",
                            "default": 3,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    }
