"""Run inside the built container; checks packaging without contacting external services."""
import json
import os
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen


class Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        path = values.get("src") if tag == "script" else values.get("href") if tag == "link" else None
        if path and path.startswith("/assets/"):
            self.paths.append(path)


def main():
    assert os.getuid() == 10001, "Application must run as the non-root rag user"
    for filename in ("/app/.env", "/app/AGENTS.md", "/app/output", "/app/frontend/node_modules"):
        assert not Path(filename).exists(), f"Unexpected local file in image: {filename}"
    base = "http://127.0.0.1:8000"
    deadline = time.monotonic() + 45
    while True:
        try:
            with urlopen(base + "/api/v1/health/live", timeout=2) as response:
                assert response.status == 200
                assert json.load(response)["status"] == "alive"
            break
        except (URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise RuntimeError("Container API did not become live") from None
            time.sleep(1)
    with urlopen(base + "/", timeout=5) as response:
        assert response.status == 200 and "text/html" in response.headers["Content-Type"]
        html = response.read().decode("utf-8")
    assets = Assets()
    assets.feed(html)
    assert assets.paths, "Built frontend assets are missing"
    for path in assets.paths:
        assert not urlsplit(path).netloc and not urlsplit(path).scheme
        with urlopen(base + path, timeout=5) as response:
            assert response.status == 200
            assert "text/html" not in response.headers["Content-Type"], "Asset fell back to index.html"
            assert response.read(), "Empty frontend asset"
    print(f"Container smoke passed: live API, UI, {len(assets.paths)} assets, non-root user, no local secrets.")


if __name__ == "__main__":
    main()
