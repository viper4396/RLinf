import asyncio

from rlinf.agents.wideseek_r2.utils.sglang_client import SGLangClient


class _FakeResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False

    async def json(self):
        return {"choices": [{"message": {"content": "judge response"}}]}


class _FakeSession:
    closed = False

    def __init__(self):
        self.request = None

    def post(self, url, json, timeout):
        self.request = {"url": url, "json": json, "timeout": timeout}
        return _FakeResponse()


def test_sglang_judge_request_enables_thinking():
    """Judge requests should enable the model's thinking chat template mode."""

    async def _run():
        session = _FakeSession()
        original_session = SGLangClient._shared_session
        SGLangClient._shared_session = session
        try:
            client = SGLangClient("judge.example", "30000", "judge-model")
            result = await client.call_sglang_api(
                [{"role": "user", "content": "judge this answer"}]
            )
        finally:
            SGLangClient._shared_session = original_session

        assert result == "judge response"
        assert session.request["json"]["chat_template_kwargs"] == {
            "enable_thinking": True
        }

    asyncio.run(_run())
