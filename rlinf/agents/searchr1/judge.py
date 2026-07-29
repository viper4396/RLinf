# Copyright 2026 The RLinf Authors.
#
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

"""OpenAI-compatible answer judge for Search-R1 evaluation."""

import asyncio
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import aiohttp
from omegaconf import DictConfig

_JUDGE_PROMPT = """Question: {question}

Labeled Answer: {correct_answer}

Predicted Answer: {predicted_answer}

Did the model give an answer equivalent to the labeled answer?

Respond with exactly "Correct" if they are equivalent, or "Incorrect" if they
are not equivalent. Do not include any other text."""


def parse_judge_response(response: str) -> float | None:
    """Parse a strict binary judge response."""
    tokens = re.sub(r"[^a-z]+", " ", response.casefold()).split()
    if "incorrect" in tokens:
        return 0.0
    if "correct" in tokens:
        return 1.0
    return None


@dataclass(frozen=True)
class JudgeRecord:
    """One Search-R1 trajectory to judge."""

    question: str
    predicted_answer: str | None
    correct_answer: Any


class SearchR1JudgeClient:
    """Concurrent client for an OpenAI-compatible chat-completions judge."""

    def __init__(self, cfg: DictConfig):
        judge_cfg = cfg.reward.judge
        endpoint = str(judge_cfg.endpoint).rstrip("/")
        self.url = (
            endpoint
            if endpoint.endswith("/v1/chat/completions")
            else f"{endpoint}/v1/chat/completions"
        )
        self.model = str(judge_cfg.model)
        self.timeout = float(judge_cfg.get("timeout", 120))
        self.max_retries = max(1, int(judge_cfg.get("max_retries", 2)))
        self.max_concurrency = max(1, int(judge_cfg.get("max_concurrency", 16)))
        self.max_tokens = max(8, int(judge_cfg.get("max_tokens", 4096)))

    async def _generate_one(
        self,
        session: aiohttp.ClientSession,
        messages: list[dict[str, str]],
        semaphore: asyncio.Semaphore,
        *,
        max_tokens: int,
    ) -> str:
        """Generate one judge response with bounded retries and concurrency."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        last_error: Exception | None = None
        async with semaphore:
            for attempt in range(self.max_retries):
                try:
                    async with session.post(
                        self.url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
                        return str(data["choices"][0]["message"]["content"])
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    last_error = error
                    if attempt + 1 < self.max_retries:
                        await asyncio.sleep(1.0)
        return f"judge request failed: {type(last_error).__name__}"

    @asynccontextmanager
    async def generator(
        self,
    ) -> AsyncIterator[Callable[[list[dict[str, str]]], Awaitable[str]]]:
        """Yield a shared OpenAI-compatible generator for structured judges."""
        connection_limit = max(32, self.max_concurrency)
        semaphore = asyncio.Semaphore(self.max_concurrency)
        connector = aiohttp.TCPConnector(
            limit=connection_limit,
            limit_per_host=connection_limit,
            enable_cleanup_closed=True,
        )
        async with aiohttp.ClientSession(
            connector=connector,
            trust_env=False,
        ) as session:

            async def generate(messages: list[dict[str, str]]) -> str:
                return await self._generate_one(
                    session,
                    messages,
                    semaphore,
                    max_tokens=self.max_tokens,
                )

            yield generate

    async def _score_one(
        self,
        session: aiohttp.ClientSession,
        record: JudgeRecord,
        semaphore: asyncio.Semaphore,
    ) -> tuple[float | None, str]:
        if record.predicted_answer is None:
            return 0.0, "missing <answer> tag"
        prompt = _JUDGE_PROMPT.format(
            question=record.question,
            correct_answer=record.correct_answer,
            predicted_answer=record.predicted_answer,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an evaluation assistant. Determine whether two "
                    "answers are equivalent."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        text = await self._generate_one(
            session,
            messages,
            semaphore,
            max_tokens=8,
        )
        return parse_judge_response(text), text

    async def score_many(
        self, records: list[JudgeRecord]
    ) -> list[tuple[float | None, str]]:
        """Judge records concurrently without using external proxy variables."""
        connection_limit = max(32, self.max_concurrency)
        semaphore = asyncio.Semaphore(self.max_concurrency)
        connector = aiohttp.TCPConnector(
            limit=connection_limit,
            limit_per_host=connection_limit,
            enable_cleanup_closed=True,
        )
        async with aiohttp.ClientSession(
            connector=connector,
            trust_env=False,
        ) as session:
            return await asyncio.gather(
                *(self._score_one(session, record, semaphore) for record in records)
            )
