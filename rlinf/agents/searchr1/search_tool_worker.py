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


import asyncio
from typing import Any

import aiohttp
from omegaconf import DictConfig

from rlinf.data.tool_call.tool_io_struct import ToolChannelRequest, ToolChannelResponse
from rlinf.scheduler import Channel
from rlinf.workers.agent.tool_worker import ToolWorker


class AsyncSearchClient:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.session: aiohttp.ClientSession | None = None
        self.semaphore: asyncio.Semaphore | None = None
        self.server_addr = self.cfg.tools.search.server_addr
        self.max_concurrency = max(
            1, int(self.cfg.tools.search.get("max_concurrency", 16))
        )
        self.max_retries = max(1, int(self.cfg.tools.search.get("max_retries", 3)))
        self.retry_delay = max(
            0.0, float(self.cfg.tools.search.get("retry_delay", 5))
        )
        print(self.server_addr)

    async def start(self):
        """Create the bounded shared HTTP client for this tool worker."""
        if self.session is not None and not self.session.closed:
            return

        connector = aiohttp.TCPConnector(
            limit=self.max_concurrency,
            limit_per_host=self.max_concurrency,
            enable_cleanup_closed=True,
        )
        self.session = aiohttp.ClientSession(connector=connector)
        self.semaphore = asyncio.Semaphore(self.max_concurrency)

    async def close(self):
        """Close the shared HTTP client and release its connections."""
        if self.session is not None and not self.session.closed:
            await self.session.close()
        self.session = None
        self.semaphore = None

    async def query_async(self, req_meta: dict[str, Any]) -> list[dict]:
        await self.start()
        assert self.session is not None
        assert self.semaphore is not None

        last_exception = None
        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    async with self.session.post(
                        f"http://{self.server_addr}/retrieve",
                        json=req_meta,
                        timeout=aiohttp.ClientTimeout(total=120, sock_connect=120),
                    ) as response:
                        response.raise_for_status()
                        res = await response.json()
                        return [
                            {
                                "documents": [r["contents"] for r in result],
                                "urls": [r["url"] for r in result],
                                "server_type": "async-search-browser",
                            }
                            for result in res["result"]
                        ]
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    last_exception = e
                    if attempt + 1 >= self.max_retries:
                        break
                    print(
                        f"Search Engine error {e}. "
                        f"Retry {attempt + 1}/{self.max_retries}."
                    )
                    await asyncio.sleep(self.retry_delay)

        raise RuntimeError(
            "Fail to post search query to RAG server"
        ) from last_exception


class SearchToolWorker(ToolWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.topk = self.cfg.tools.search.topk
        self.dummy_mode = self.cfg.tools.search.get("dummy_mode", False)
        self.request_processor_task = None
        self.active_tasks: set[asyncio.Task[Any]] = set()
        self.search_client = AsyncSearchClient(cfg=self.cfg)

    def init_worker(self, input_channel: Channel, output_channel: Channel):
        self.input_channel = input_channel
        self.output_channel = output_channel

    def start_server(self):
        loop = asyncio.get_running_loop()
        self.request_processor_task = loop.create_task(self._process_requests())

    def stop_server(self):
        """Cancel active requests and close the shared HTTP client."""
        if self.request_processor_task and not self.request_processor_task.done():
            self.request_processor_task.cancel()
        for task in list(self.active_tasks):
            task.cancel()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._shutdown())

    async def _shutdown(self):
        """Wait for cancelled requests and release HTTP resources."""
        tasks = list(self.active_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.search_client.close()

    async def _process_requests(self):
        if not self.dummy_mode:
            await self.search_client.start()

        def process_tool_result(response):
            res = {}
            res["documents"] = response[0]["documents"]
            res["urls"] = response[0]["urls"]
            res["type"] = "search"

            documents = response[0]["documents"][: self.topk]
            urls = res["urls"][: self.topk]
            if len(documents) > 0:
                doc_id_template = "[Doc {doc_id}]({url}):\n"
                text = (
                    "<information>\n"
                    + "\n\n".join(
                        [
                            doc_id_template.format(doc_id=str(k + 1), url=url)
                            + doc[:5000]
                            for k, (doc, url) in enumerate(zip(documents, urls))
                        ]
                    )
                    + "\n</information>"
                )
            else:
                text = "<information>\nNo search results are found.\n</information>"

            full_text = "\n\n" + text + "\n<think>\n"
            return full_text

        async def generate_and_send(channel_key: str, tool_args: dict):
            try:
                req_meta = {
                    "queries": [tool_args["keyword"]],
                    "topk": self.topk,
                    "return_scores": False,
                }
                full_text = (
                    "<information>\nNo search results are found.\n</information>"
                )
                if not self.dummy_mode:
                    response = await self.search_client.query_async(req_meta)
                    full_text = process_tool_result(response)

                result = ToolChannelResponse(
                    success=True,
                    result=full_text,
                )
            except Exception as e:
                result = ToolChannelResponse(
                    success=False,
                    result=e,
                )
            await self.output_channel.put(
                result, key=channel_key, async_op=True
            ).async_wait()
            # self.logger.info("SearchToolWorker._process_requests: sent response")

        while True:
            request: ToolChannelRequest = await self.input_channel.get(
                async_op=True
            ).async_wait()
            # self.logger.info("SearchToolWorker._process_requests: got request")
            assert request.request_type == "execute", (
                "SearchToolWorker has no session, so only get 'execute' request_type"
            )
            assert request.tool_name == "search", (
                "SearchToolWorker only execute tool_name 'search'"
            )
            task = asyncio.create_task(
                generate_and_send(request.session_id, request.tool_args)
            )
            self.active_tasks.add(task)
            task.add_done_callback(self.active_tasks.discard)
