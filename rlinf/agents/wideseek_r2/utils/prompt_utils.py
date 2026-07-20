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

from rlinf.agents.wideseek_r2.utils.prompt import (
    BOXED_FORMAT_EN,
    MARKDOWN_FORMAT_EN,
    MARKDOWN_LIST_FORMAT_EN,
    MARKDOWN_SET_FORMAT_EN,
    PLANNER_ITEM_STRATEGY_EN,
    PLANNER_LIST_STRATEGY_EN,
    PLANNER_SET_STRATEGY_EN,
    PLANNER_TABLE_STRATEGY_EN,
    SINGLE_AGENT_ITEM_STRATEGY_EN,
    SINGLE_AGENT_LIST_STRATEGY_EN,
    SINGLE_AGENT_SET_STRATEGY_EN,
    SINGLE_AGENT_TABLE_STRATEGY_EN,
    SYSTEM_PROMPT_PLANNER,
    SYSTEM_PROMPT_PLANNER_NOSHOT,
    SYSTEM_PROMPT_PLANNER_UNLIMITED,
    SYSTEM_PROMPT_SINGLE_AGENT,
    SYSTEM_PROMPT_SINGLE_AGENT_NOSHOT,
    SYSTEM_PROMPT_WORKER,
    USER_PROMPT_PLANNER,
    USER_PROMPT_SINGLE_AGENT,
    USER_PROMPT_WORKER,
)
from rlinf.agents.wideseek_r2.utils.tool_description import get_tools_description

_PLANNER_STRATEGIES = {
    "item": PLANNER_ITEM_STRATEGY_EN,
    "set": PLANNER_SET_STRATEGY_EN,
    "list": PLANNER_LIST_STRATEGY_EN,
    "table": PLANNER_TABLE_STRATEGY_EN,
}
_SINGLE_AGENT_STRATEGIES = {
    "item": SINGLE_AGENT_ITEM_STRATEGY_EN,
    "set": SINGLE_AGENT_SET_STRATEGY_EN,
    "list": SINGLE_AGENT_LIST_STRATEGY_EN,
    "table": SINGLE_AGENT_TABLE_STRATEGY_EN,
}


def _format_instruction(answer_mode: str, answer_type: str | None = None) -> str:
    """Return the final-answer format instruction for the given answer mode.

    Args:
        answer_mode: Either ``"markdown"`` or ``"boxed"``.
        answer_type: Optional answer structure used to specialize Markdown
            collection formatting.

    Returns:
        The format instruction string injected into the system prompt.

    Raises:
        ValueError: If ``answer_mode`` is not a supported value.
    """
    if answer_mode == "markdown":
        normalized_type = (
            str(answer_type).strip().lower() if answer_type is not None else None
        )
        if normalized_type == "set":
            return MARKDOWN_SET_FORMAT_EN
        if normalized_type == "list":
            return MARKDOWN_LIST_FORMAT_EN
        return MARKDOWN_FORMAT_EN
    if answer_mode == "boxed":
        return BOXED_FORMAT_EN
    raise ValueError(
        f"Unsupported answer_mode {answer_mode!r}; expected 'markdown' or 'boxed'."
    )


def _answer_type_strategy(
    answer_type: str | None,
    answer_mode: str,
    role: str,
) -> str:
    """Return the role-specific strategy injected into a no-shot prompt.

    ``answer_type`` normally comes from the dataset. The fallback mirrors the
    dataset contract for direct callers of this prompt utility: boxed answers
    are items, while Markdown answers are tables.

    Args:
        answer_type: One of ``item``, ``set``, ``list``, or ``table``.
        answer_mode: Final-answer wrapper mode used when ``answer_type`` is None.
        role: Either ``planner`` or ``single``.

    Returns:
        The strategy text for the selected role and answer type.

    Raises:
        ValueError: If the role or answer type is unsupported.
    """
    if role == "planner":
        strategies = _PLANNER_STRATEGIES
    elif role == "single":
        strategies = _SINGLE_AGENT_STRATEGIES
    else:
        raise ValueError(f"Unsupported strategy role {role!r}.")

    if answer_type is None:
        normalized_type = "item" if answer_mode == "boxed" else "table"
    else:
        normalized_type = str(answer_type).strip().lower()
    if normalized_type not in strategies:
        supported = ", ".join(strategies)
        raise ValueError(
            f"Unsupported answer_type {answer_type!r}; expected one of {supported}."
        )
    return strategies[normalized_type]


def _fanout_guidance(max_workers_per_planner: int) -> str:
    """Return the per-turn fan-out guidance sentence for the planner prompt.

    Mirrors the ``create_sub_agents`` tool description: a negative value means
    unlimited parallel sub-agents, a non-negative value caps the count.
    """
    if max_workers_per_planner < 0:
        return (
            "There is NO limit on the number of sub-agents you may launch in a "
            "single call: whenever a turn requires researching many independent "
            "items, create one sub-agent per item and launch them all in parallel "
            "in the same turn."
        )
    return (
        "You may launch at most "
        f"{max_workers_per_planner} sub-agents in a single call, so when a turn "
        "requires more independent items than that, delegate as many as allowed "
        "now and handle the remaining ones in the following turns."
    )


def get_prompt_planner(
    question: str,
    answer_mode: str,
    add_few_shot: bool,
    max_workers_per_planner: int,
    answer_type: str | None = None,
) -> list:
    """Build the planner prompt with optional few-shot examples.

    When ``max_workers_per_planner`` is negative the planner is uncapped and the
    unlimited-mode few-shot is used; otherwise the capped few-shot injects the
    limit so the narrative matches the parser's enforcement.
    """
    format_instruction = _format_instruction(answer_mode, answer_type)
    if add_few_shot:
        if max_workers_per_planner < 0:
            system = SYSTEM_PROMPT_PLANNER_UNLIMITED.format(format_instruction)
        else:
            system = SYSTEM_PROMPT_PLANNER.format(
                format_instruction, max_workers_per_planner=max_workers_per_planner
            )
    else:
        system = SYSTEM_PROMPT_PLANNER_NOSHOT.format(
            format_instruction,
            fanout_guidance=_fanout_guidance(max_workers_per_planner),
            answer_type_strategy=_answer_type_strategy(
                answer_type, answer_mode, role="planner"
            ),
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": USER_PROMPT_PLANNER.format(question)},
    ]


def get_prompt_worker(origin_question: str, subtask: str) -> list:
    """Build the worker prompt for a single subtask."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_WORKER},
        {
            "role": "user",
            "content": USER_PROMPT_WORKER.format(origin_question, subtask),
        },
    ]


def get_prompt_single_agent(
    question: str,
    answer_mode: str,
    add_few_shot: bool,
    answer_type: str | None = None,
) -> list:
    """Build the single-agent prompt with optional few-shot examples."""
    format_instruction = _format_instruction(answer_mode, answer_type)
    if add_few_shot:
        system = SYSTEM_PROMPT_SINGLE_AGENT.format(format_instruction)
    else:
        system = SYSTEM_PROMPT_SINGLE_AGENT_NOSHOT.format(
            format_instruction,
            answer_type_strategy=_answer_type_strategy(
                answer_type, answer_mode, role="single"
            ),
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": USER_PROMPT_SINGLE_AGENT.format(question)},
    ]


def build_message_history_and_tools(
    origin_question: str,
    role: str,
    answer_mode: str,
    add_few_shot: bool,
    max_workers_per_planner: int,
    max_toolcall_per_worker: int,
    main_task: str | None = None,
    answer_type: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Build role-specific prompt history and exposed tool descriptions.

    Args:
        origin_question: Query text for this role loop.
        role: Current role (`planner`, `worker`, or `single`).
        answer_mode: Answer mode for this sample (``markdown`` or ``boxed``).
        add_few_shot: Whether to include few-shot examples in the system prompt.
        max_workers_per_planner: Sub-agent per-call limit for prompt/tool
            descriptions; negative means unlimited.
        max_toolcall_per_worker: Search/access per-call limit for tool descriptions.
        main_task: Parent task text required for worker prompts.
        answer_type: Dataset-provided answer structure for no-shot main roles.

    Returns:
        A tuple of `(message_history, tools)` for chat-template rendering.
    """
    tools_description = get_tools_description(
        max_workers_per_planner=max_workers_per_planner,
        max_toolcall_per_worker=max_toolcall_per_worker,
    )
    if role == "planner":
        message_history = get_prompt_planner(
            origin_question,
            answer_mode=answer_mode,
            add_few_shot=add_few_shot,
            max_workers_per_planner=max_workers_per_planner,
            answer_type=answer_type,
        )
        tools = [tools_description["create_sub_agents"]]
    elif role == "worker":
        assert main_task is not None, "Worker must have main_task provided"
        message_history = get_prompt_worker(main_task, origin_question)
        tools = [tools_description["search"], tools_description["access"]]
    elif role == "single":
        message_history = get_prompt_single_agent(
            origin_question,
            answer_mode=answer_mode,
            add_few_shot=add_few_shot,
            answer_type=answer_type,
        )
        tools = [
            tools_description["search_single_agent"],
            tools_description["access_single_agent"],
        ]
    else:
        raise ValueError(f"Invalid role: {role}")
    return message_history, tools


def get_access_summary_messages(info_to_extract, page_content):
    """Build extraction messages that summarize accessed page content."""
    system_prompt = (
        "You are an information extraction assistant.\n"
        "You MUST base your output ONLY on the provided webpage content.\n"
        "You are strictly forbidden from using any prior knowledge, assumptions, or external information.\n\n"
        "Your task is NOT to answer the question directly, but to extract and summarize all information from the webpage that is relevant to the specified information requirement.\n\n"
        "If the webpage does NOT contain the exact requested information:\n"
        "- Extract the most closely related information from the webpage and explain its relevance.\n"
        '- If there is truly nothing related, explicitly state: "This webpage contains no information relevant to the request."\n\n'
        "You must NOT hallucinate, infer, or guess.\n"
        "You must NOT answer from your own knowledge.\n\n"
        "Your output MUST be a clear, complete, and well-structured summary report.\n"
        "The report should:\n"
        "- Be organized with headings or bullet points when appropriate\n"
        "- Include concrete facts, statements, or quotations from the webpage as evidence\n"
        "- Focus exclusively on information relevant to the request\n"
        "- Exclude any general summaries or unrelated content\n"
        "- Exclude any meta-commentary about your process\n"
    )

    user_prompt = (
        f"INFORMATION TO EXTRACT:\n{info_to_extract}\n\n"
        f"CONTENT TO ANALYZE:\n{page_content}\n\n"
        "Extract and summarize only the information relevant to the request above.\n"
        "Follow all instructions strictly."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def get_first_turn_hint(max_turns: int) -> str:
    """Return the hint appended to the first user turn."""
    return (
        "\n\nThis is your first turn to answer the question. "
        f"You must finish your answer within {max_turns} turns"
    )


def get_next_turn_hint(next_turn_idx: int, max_turns: int) -> str:
    """Return the hint appended after a tool response."""
    return (
        f"\n\nYour next answer will be on turn {next_turn_idx}. "
        f"You MUST finish the entire answer by turn {max_turns}."
    )


def get_planner_subtask_result_message(
    subtask_idx: int,
    subtask_text: str,
    worker_summary: str,
) -> str:
    """Format a successful subtask result for the planner."""
    return f"# Subtask {subtask_idx}:\n{subtask_text}\n# Result:\n{worker_summary}"


def get_planner_subtask_failed_message(
    subtask_idx: int,
    subtask_text: str,
) -> str:
    """Format a failed subtask result for the planner."""
    return (
        f"# Subtask {subtask_idx}:\n{subtask_text}\n# Result:\n"
        "The current subagent did not return a valid final answer "
        "(no <answer>...</answer> block was produced) for this subtask. "
        "Please retry."
    )


def get_search_tool_message(query: str, search_result: str) -> str:
    """Format a search tool response."""
    return f"# Search query:\n{query}\n# Result:\n{search_result}"


def get_access_tool_message(url: str, page_content: str) -> str:
    """Format an access tool response."""
    return f"# Access URL:\n{url}\n# Result:\n{page_content}"


def get_access_summary_tool_message(
    url: str,
    info_to_extract: str | None,
    summary: str,
) -> str:
    """Format a summarized access tool response."""
    return (
        f"# Access URL:\n{url}\n# Info to extract:\n{info_to_extract}\n"
        f"# Result:\n{summary}"
    )
