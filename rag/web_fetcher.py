"""Safe, best-effort fetching and readable text extraction for web pages."""

from __future__ import annotations

import html
import logging
import re
from html.parser import HTMLParser
from urllib.parse import urlparse

import requests

LOGGER = logging.getLogger(__name__)
DEFAULT_TIMEOUT = 12
MAX_PAGE_BYTES = 4_000_000
USER_AGENT = "Mozilla/5.0 (compatible; TruthBridge/1.0; +https://github.com/truthbridge)"
BOT_MARKERS = (
    "captcha",
    "verify you are human",
    "are you a robot",
    "access denied",
    "just a moment",
    "checking your browser",
    "enable javascript and cookies",
    "automated access",
    "too many requests",
)


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"script", "style", "nav", "footer", "header", "noscript", "svg"}:
            self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in {"script", "style", "nav", "footer", "header", "noscript", "svg"}:
            self.skip_depth = max(0, self.skip_depth - 1)

    def handle_data(self, data):
        if not self.skip_depth:
            value = html.unescape(data).strip()
            if value:
                self.parts.append(value)


def _readable_text(markup: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(markup, "html.parser")
        for element in soup(
            ["script", "style", "nav", "footer", "header", "noscript", "svg",
             "form", "aside", "title", "meta", "link", "h1", "h2", "h3", "h4", "h5", "h6"]
        ):
            element.decompose()
        main = soup.find("article") or soup.find("main") or soup.body or soup
        paragraphs = []
        for element in main.find_all(["p", "blockquote"]):
            value = re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()
            if (
                len(value) >= 45
                and not value.endswith("?")
                and not re.match(r"^(share|subscribe|follow|read more|sign up)\b", value, re.I)
            ):
                paragraphs.append(value)
        if paragraphs:
            return "\n\n".join(dict.fromkeys(paragraphs))
        return ""
    except ImportError:
        parser = _TextParser()
        parser.feed(markup)
        return " ".join(parser.parts)


def _is_bot_or_boilerplate(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    if len(normalized) < 80:
        return True
    if any(marker in normalized for marker in BOT_MARKERS):
        return True
    # Pages that contain only repeated navigation/consent text are not
    # evidence even when they exceed the minimum character count.
    words = normalized.split()
    return len(words) >= 20 and len(set(words)) / len(words) < 0.12


def fetch_webpage(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Fetch a HTTP(S) page and return readable text, or ``""`` on any skip."""

    parsed = urlparse(url or "")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        LOGGER.warning("Skipping invalid web URL: %r", url)
        return ""
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            allow_redirects=True,
        )
        if response.status_code in {401, 403, 404}:
            LOGGER.info("Skipping inaccessible page %s (HTTP %d)", url, response.status_code)
            return ""
        if response.status_code >= 400:
            LOGGER.info("Skipping failed page %s (HTTP %d)", url, response.status_code)
            return ""
        response.raise_for_status()
        response_url = getattr(response, "url", url) or url
        final = urlparse(response_url)
        if final.scheme not in {"http", "https"} or not final.netloc:
            LOGGER.info("Skipping unsafe redirect %s -> %s", url, response_url)
            return ""
        if response.history:
            LOGGER.info("Followed %d redirect(s): %s -> %s", len(response.history), url, response_url)
        content_type = response.headers.get("content-type", "").lower()
        if content_type and not any(kind in content_type for kind in ("text/html", "application/xhtml", "text/plain")):
            LOGGER.info("Skipping non-HTML page %s (%s)", url, content_type)
            return ""
        if len(response.content) > MAX_PAGE_BYTES:
            LOGGER.info("Skipping oversized page %s", url)
            return ""
        text = "\n\n".join(
            re.sub(r"[ \t]+", " ", paragraph).strip()
            for paragraph in _readable_text(response.text).split("\n\n")
            if paragraph.strip()
        )
        if _is_bot_or_boilerplate(text):
            LOGGER.info("Skipping empty/boilerplate page %s", url)
            return ""
        LOGGER.info("Fetched page url=%s chars=%d", url, len(text))
        return text
    except requests.Timeout:
        LOGGER.warning("Fetch timed out url=%s", url)
        return ""
    except requests.RequestException as exc:
        LOGGER.warning("Fetch failed url=%s error=%s", url, exc)
        return ""
    except Exception as exc:
        LOGGER.warning("Page extraction failed url=%s error=%s", url, exc)
        return ""