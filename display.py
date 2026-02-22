import html as html_lib
import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from openai import APITimeoutError, OpenAI


load_dotenv()

DISPLAY_MODEL = os.getenv("DISPLAY_MODEL", "gpt-4.1-nano")
DISPLAY_TIMEOUT_SECONDS_RAW = os.getenv("DISPLAY_TIMEOUT_SECONDS")
DISPLAY_TEMPERATURE = float(os.getenv("DISPLAY_TEMPERATURE", "0.1"))
DISPLAY_MAX_OUTPUT_TOKENS = int(os.getenv("DISPLAY_MAX_OUTPUT_TOKENS", "900"))
DISPLAY_MAX_RETRIES = int(os.getenv("DISPLAY_MAX_RETRIES", "0"))
DISPLAY_ALLOW_CHAT_FALLBACK = os.getenv("DISPLAY_ALLOW_CHAT_FALLBACK", "0") == "1"

SYSTEM_PROMPT = """You are an HTML screen editor.
Return only raw HTML text. Never return markdown, code fences, explanations, or JSON.

You receive:
1) A user request describing a screen change.
2) The current HTML on screen.

Task:
- Update the current HTML to satisfy the user request.
- If current HTML is empty, create a minimal valid HTML document.
- Keep unrelated content intact when possible.
- Use plain HTML/CSS only. No external assets, no JS frameworks, no script tags.
- Build visuals directly in HTML/CSS/SVG.
- Visual style rule: for simple shapes, charts, symbols, and diagrams, use clean SVG/HTML/CSS.
- Visual style rule: for complex real-world things (animals, people, vehicles, detailed scenes), prefer a large emoji representation.
- Never use remote URLs, image links, `<img>` tags, CSS `background-image: url(...)`, or external font/link imports.
- Ensure the result is valid and renderable.
"""


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")
    timeout = (
        float(DISPLAY_TIMEOUT_SECONDS_RAW)
        if DISPLAY_TIMEOUT_SECONDS_RAW and DISPLAY_TIMEOUT_SECONDS_RAW.strip()
        else None
    )
    return OpenAI(
        api_key=api_key,
        timeout=timeout,
        max_retries=DISPLAY_MAX_RETRIES,
    )


def _strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def _extract_output_text(response) -> str:
    direct = getattr(response, "output_text", "")
    if direct:
        return direct

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _has_disallowed_assets(html: str) -> bool:
    return any(
        re.search(pattern, html, flags=re.IGNORECASE)
        for pattern in [
            r"<img\b",
            r"src\s*=\s*['\"]https?://",
            r"url\(\s*['\"]?\s*https?://",
            r"@import\s+url\(",
            r"<link\b[^>]*href\s*=\s*['\"]https?://",
        ]
    )


def _generate_html_with_responses(user_request: str, current_html: str) -> str:
    response = _get_client().responses.create(
        model=DISPLAY_MODEL,
        temperature=DISPLAY_TEMPERATURE,
        max_output_tokens=DISPLAY_MAX_OUTPUT_TOKENS,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "User request:\n"
                            f"{user_request.strip()}\n\n"
                            "Current HTML:\n"
                            f"{current_html.strip() or '<empty>'}"
                        ),
                    }
                ],
            },
        ],
    )
    return _extract_output_text(response)


def _generate_html_with_chat(user_request: str, current_html: str) -> str:
    response = _get_client().chat.completions.create(
        model=DISPLAY_MODEL,
        temperature=DISPLAY_TEMPERATURE,
        max_tokens=DISPLAY_MAX_OUTPUT_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "User request:\n"
                    f"{user_request.strip()}\n\n"
                    "Current HTML:\n"
                    f"{current_html.strip() or '<empty>'}"
                ),
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def screen(user_request: str, current_html: str) -> str:
    """
    Convert a natural-language display request into updated screen HTML.

    Args:
        user_request: The user instruction (for example: "I want half the screen red").
        current_html: The HTML currently on the screen.

    Returns:
        Updated HTML as a raw string.
    """
    if not user_request or not user_request.strip():
        raise ValueError("user_request must be a non-empty string.")

    existing_html = current_html or ""

    try:
        html = _generate_html_with_responses(user_request, current_html or "")
    except APITimeoutError:
        raise RuntimeError(
            "Display generation timed out before response. "
            "Set DISPLAY_TIMEOUT_SECONDS to a larger value, or unset it to allow long requests."
        )
    except Exception as first_error:
        if DISPLAY_ALLOW_CHAT_FALLBACK:
            html = _generate_html_with_chat(user_request, existing_html)
        else:
            raise RuntimeError(
                "Display generation failed on Responses API. "
                "Set DISPLAY_ALLOW_CHAT_FALLBACK=1 to try chat fallback."
            ) from first_error

    html = _strip_markdown_fences(html)
    if not html:
        raise RuntimeError("Model returned an empty response.")
    if _has_disallowed_assets(html):
        raise RuntimeError(
            "Model output used disallowed external assets. "
            "Only inline HTML/CSS/SVG visuals are allowed."
        )
    return html


def wrap_screen(user_request: str, current_html: str) -> str:
    """Alias for screen() used by the web app integration."""
    return screen(user_request, current_html)


def render_error_html(error_message: str) -> str:
    """Return a small inline HTML error page for display fallback."""
    escaped = html_lib.escape(error_message)
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>"
        "<title>Display Error</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:0;padding:16px;background:#fff7f6;color:#5c1d18;}"
        "h2{margin:0 0 10px 0;font-size:18px;}"
        "pre{white-space:pre-wrap;background:#fff;border:1px solid #f0c8c1;padding:10px;border-radius:8px;}"
        "</style>"
        "</head><body>"
        "<h2>Display generation failed</h2>"
        f"<pre>{escaped}</pre>"
        "</body></html>"
    )
