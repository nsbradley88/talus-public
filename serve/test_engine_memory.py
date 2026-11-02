"""#40: an engine's memory refusal (ERR insufficient_memory: ...) is a 503 with code insufficient_memory, and the
next request is served.

    python -m unittest serve.test_engine_memory -v
"""
import http.client
import json
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineMemory, MockEngine, Service, engine_error, serve

ROOT = Path(__file__).resolve().parents[1]
MSG = ("insufficient_memory: host MemAvailable 7900 MiB is below the 8192 MiB floor at request start after shedding "
       "rebuildable caches (decode graphs 40 MiB); this request was not completed, the server keeps running - retry "
       "when memory is free")


class RefuseOnce(MockEngine):
    """The first request is refused for memory, as a real engine's ERR line would be; later ones answer."""

    def __init__(self, tok):
        super().__init__(tok, "ok")
        self.refused = False

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        if not self.refused:
            self.refused = True
            raise engine_error(MSG)
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class EngineMemoryTest(unittest.TestCase):
    def test_mapping(self):
        self.assertIsInstance(engine_error(MSG), EngineMemory)
        self.assertNotIsInstance(engine_error("bad prompt ids"), EngineMemory)

    def test_503_then_served(self):
        tok = ByteTokenizer()
        svc = Service(RefuseOnce(tok), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            def post():
                c = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=10)
                try:
                    c.request("POST", "/v1/chat/completions",
                              json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}),
                              {"Content-Type": "application/json"})
                    r = c.getresponse()
                    return r.status, json.loads(r.read())
                finally:
                    c.close()
            status, body = post()
            self.assertEqual(status, 503, body)
            self.assertEqual(body["error"]["code"], "insufficient_memory")
            self.assertEqual(body["error"]["type"], "server_error")
            self.assertIn("the server keeps running", body["error"]["message"])
            status, body = post()
            self.assertEqual(status, 200, body)
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
