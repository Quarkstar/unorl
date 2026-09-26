"""Check rendered local links, figures and downloads under a Pages base path."""

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = set()

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in {"src", "href"} and value:
                self.urls.add(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2] / "_build/html"
    failures = set()
    count = 0
    pages = list(root.rglob("*.html"))
    if not pages:
        raise SystemExit("No rendered HTML found; build the book first.")
    for page in pages:
        links = Links()
        links.feed(page.read_text())
        for url in links.urls:
            parsed = urlsplit(url)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            path = unquote(parsed.path)
            if path.startswith("/"):
                prefix = args.base_url.rstrip("/")
                if prefix and not (path == prefix or path.startswith(prefix + "/")):
                    failures.add(f"{page.name}: URL outside configured base path: {url}")
                    continue
                path = path[len(prefix) :].lstrip("/")
                target = root / path
            else:
                target = page.parent / path
            count += 1
            if not any(
                p.exists() for p in [target, Path(str(target) + ".html"), target / "index.html"]
            ):
                failures.add(f"{page.name}: missing target {url}")
    if failures:
        raise SystemExit("\n".join(sorted(failures)))
    print(f"Checked {len(pages)} HTML pages and {count} local references.")


if __name__ == "__main__":
    main()
