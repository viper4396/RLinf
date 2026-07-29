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

"""Serper search with optional Jina page enrichment for Search-R1."""

import asyncio
import os
import random
from typing import Any

import aiohttp
from omegaconf import DictConfig


class SearchR1OnlineSearchClient:
    """Search the web with Serper and enrich selected results through Jina."""

    def __init__(self, cfg: DictConfig):
        search_cfg = cfg.tools.search
        self.topk = int(search_cfg.get("topk", 5))
        self.jina_topk = int(search_cfg.get("jina_topk", 2))
        self.jina_page_chars = int(search_cfg.get("jina_page_chars", 6000))
        self.max_retries = max(1, int(search_cfg.get("max_retries", 3)))
        self.retry_delay = max(0.0, float(search_cfg.get("retry_delay", 1.0)))
        self.search_oversample = max(0, int(search_cfg.get("search_oversample", 5)))
        self.blocked_patterns = tuple(
            str(pattern).casefold()
            for pattern in search_cfg.get("blocked_url_patterns", [])
            if str(pattern).strip()
        )
        self.use_jina = bool(search_cfg.get("use_jina", True))

        self.serper_api_key = os.environ.get("SERPER_API_KEY", "")
        self.jina_api_key = os.environ.get("JINA_API_KEY", "")
        if not self.serper_api_key:
            raise RuntimeError("SERPER_API_KEY is required for online Search-R1")
        if self.use_jina and not self.jina_api_key:
            raise RuntimeError(
                "JINA_API_KEY is required when tools.search.use_jina is enabled"
            )

        self.search_semaphore = asyncio.Semaphore(
            max(1, int(search_cfg.get("max_concurrency", 20)))
        )
        self.jina_semaphore = asyncio.Semaphore(
            max(1, int(search_cfg.get("jina_max_concurrency", 10)))
        )
        self.session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        """Create the proxy-aware shared HTTP session."""
        if self.session is not None and not self.session.closed:
            return
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=50,
            ttl_dns_cache=600,
            enable_cleanup_closed=True,
        )
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=180, sock_connect=60),
            trust_env=True,
        )

    async def close(self) -> None:
        """Close the HTTP session."""
        if self.session is not None and not self.session.closed:
            await self.session.close()
        self.session = None

    def _is_blocked(self, result: dict[str, Any]) -> bool:
        haystack = " ".join(
            str(result.get(key, "")) for key in ("link", "title", "snippet")
        ).casefold()
        return any(pattern in haystack for pattern in self.blocked_patterns)

    async def _request_json(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> dict[str, Any]:
        await self.start()
        assert self.session is not None
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    response.raise_for_status()
                    return await response.json()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                last_error = error
                if attempt + 1 < self.max_retries:
                    await asyncio.sleep(
                        self.retry_delay * (2**attempt) + random.uniform(0.0, 0.5)
                    )
        raise RuntimeError(f"online search request failed: {url}") from last_error

    async def _serper_search(self, query: str, topk: int) -> list[dict[str, Any]]:
        async with self.search_semaphore:
            data = await self._request_json(
                "POST",
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": self.serper_api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "q": query[:2000],
                    "num": topk + self.search_oversample,
                },
            )
        return [
            result for result in data.get("organic", []) if not self._is_blocked(result)
        ][:topk]

    async def _jina_access(self, url: str) -> str:
        if not self.use_jina:
            return ""
        await self.start()
        assert self.session is not None
        last_error: Exception | None = None
        async with self.jina_semaphore:
            for attempt in range(self.max_retries):
                try:
                    async with self.session.get(
                        f"https://r.jina.ai/{url}",
                        headers={"Authorization": f"Bearer {self.jina_api_key}"},
                        timeout=aiohttp.ClientTimeout(total=120, sock_connect=60),
                    ) as response:
                        response.raise_for_status()
                        return (await response.text())[: self.jina_page_chars]
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    last_error = error
                    if attempt + 1 < self.max_retries:
                        await asyncio.sleep(
                            self.retry_delay * (2**attempt) + random.uniform(0.0, 0.5)
                        )
        return f"Jina page access failed: {type(last_error).__name__}"

    async def query_async(self, req_meta: dict[str, Any]) -> list[dict[str, Any]]:
        """Return Search-R1-compatible documents and URLs for each query."""
        queries = [str(query) for query in req_meta.get("queries", [])]
        topk = int(req_meta.get("topk", self.topk))
        search_results = await asyncio.gather(
            *(self._serper_search(query, topk) for query in queries)
        )

        formatted: list[dict[str, Any]] = []
        for results in search_results:
            urls = [str(result.get("link", "")) for result in results]
            pages = [""] * len(results)
            page_results = await asyncio.gather(
                *(self._jina_access(url) for url in urls[: self.jina_topk])
            )
            pages[: len(page_results)] = page_results

            documents = []
            for result, page in zip(results, pages):
                summary = " ".join(
                    part
                    for part in (
                        str(result.get("title", "")).strip(),
                        str(result.get("snippet", "")).strip(),
                    )
                    if part
                )
                documents.append(
                    f"{summary}\n\n[Jina page]\n{page}" if page else summary
                )
            formatted.append(
                {
                    "documents": documents,
                    "urls": urls,
                    "server_type": "serper-jina",
                }
            )
        return formatted
