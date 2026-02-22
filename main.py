#!/usr/bin/env python3
"""Flask app for speech transcription controls and visual display updates."""

from __future__ import annotations

import atexit
import json
import queue
import threading
from collections import deque

from flask import Response, Flask, redirect, render_template, request, stream_with_context, url_for

from display import render_error_html, wrap_screen
from speech_text import SpeechToTextWorker


app = Flask(__name__)
MAX_TRANSCRIPT_LINES = 400
DISPLAY_COMMAND_QUEUE_MAX = 48
DISPLAY_EVENT_HISTORY_MAX = 120


class TranscriptStore:
    """Thread-safe transcript buffer with incremental event ids."""

    def __init__(self, max_lines: int) -> None:
        self._lines: deque[tuple[int, str]] = deque(maxlen=max_lines)
        self._next_id = 0
        self._cond = threading.Condition()

    def append(self, line: str) -> None:
        with self._cond:
            self._next_id += 1
            self._lines.append((self._next_id, line))
            self._cond.notify_all()

    def snapshot(self) -> tuple[list[str], int]:
        with self._cond:
            lines = [line for _, line in self._lines]
            latest_id = self._next_id
        return lines, latest_id

    def wait_for_pending(self, cursor: int, timeout: float) -> tuple[int, list[tuple[int, str]]]:
        with self._cond:
            if self._lines and cursor < self._lines[0][0]:
                cursor = self._lines[0][0] - 1

            has_new = self._cond.wait_for(lambda: self._next_id > cursor, timeout=timeout)
            if not has_new:
                return cursor, []

            pending = [(event_id, line) for event_id, line in self._lines if event_id > cursor]
            return cursor, pending


class DisplayEngine:
    """Background display generator isolated from transcription threads."""

    def __init__(self, max_queue: int, max_events: int) -> None:
        self._commands: queue.Queue[str] = queue.Queue(maxsize=max_queue)
        self._events: deque[tuple[int, dict[str, str]]] = deque(maxlen=max_events)
        self._next_id = 0
        self._current_html = ""
        self._cond = threading.Condition()
        self._shutdown_event = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True, name="display-worker")
        self._worker.start()

    def submit(self, command: str) -> None:
        normalized = command.strip()
        if not normalized:
            return
        try:
            self._commands.put_nowait(normalized)
        except queue.Full:
            # Never block transcript callback path; drop oldest command when overloaded.
            try:
                self._commands.get_nowait()
            except queue.Empty:
                pass
            try:
                self._commands.put_nowait(normalized)
            except queue.Full:
                pass

    def snapshot(self) -> tuple[str, int]:
        with self._cond:
            return self._current_html, self._next_id

    def wait_for_pending(
        self,
        cursor: int,
        timeout: float,
    ) -> tuple[int, list[tuple[int, dict[str, str]]]]:
        with self._cond:
            if self._events and cursor < self._events[0][0]:
                cursor = self._events[0][0] - 1

            has_new = self._cond.wait_for(
                lambda: self._next_id > cursor or self._shutdown_event.is_set(),
                timeout=timeout,
            )
            if not has_new or self._next_id <= cursor:
                return cursor, []

            pending = [(event_id, payload) for event_id, payload in self._events if event_id > cursor]
            return cursor, pending

    def shutdown(self) -> None:
        self._shutdown_event.set()
        with self._cond:
            self._cond.notify_all()
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)

    def _push_event(self, payload: dict[str, str]) -> None:
        with self._cond:
            self._next_id += 1
            html = payload.get("html")
            if html is not None:
                self._current_html = html
            self._events.append((self._next_id, payload))
            self._cond.notify_all()

    def _run(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                command = self._commands.get(timeout=0.2)
            except queue.Empty:
                continue

            current_html, _ = self.snapshot()
            self._push_event({"status": "generating", "command": command})

            try:
                updated_html = wrap_screen(command, current_html)
            except Exception as exc:  # pragma: no cover - runtime guard
                message = str(exc)
                self._push_event(
                    {
                        "status": "error",
                        "command": command,
                        "error": message,
                        "html": render_error_html(message),
                    }
                )
                continue

            self._push_event(
                {
                    "status": "ready",
                    "command": command,
                    "html": updated_html,
                }
            )


transcripts = TranscriptStore(MAX_TRANSCRIPT_LINES)
display_engine = DisplayEngine(
    max_queue=DISPLAY_COMMAND_QUEUE_MAX,
    max_events=DISPLAY_EVENT_HISTORY_MAX,
)


def parse_command_line(line: str) -> str | None:
    prefix = 'COMMAND FOUND: "'
    if line.startswith(prefix) and line.endswith('"'):
        return line[len(prefix) : -1]
    return None


def on_transcript_line(line: str) -> None:
    transcripts.append(line)
    command = parse_command_line(line)
    if command is not None:
        display_engine.submit(command)


stt_worker = SpeechToTextWorker(transcript_callback=on_transcript_line)
atexit.register(stt_worker.shutdown)
atexit.register(display_engine.shutdown)


@app.get("/")
def index() -> str:
    is_listening = stt_worker.status == "listening"
    initial_lines_raw, latest_event_id = transcripts.snapshot()
    initial_lines = []
    latest_command = None
    for line in initial_lines_raw:
        parsed = parse_command_line(line)
        if parsed is None:
            initial_lines.append(line)
        else:
            latest_command = parsed

    initial_display_html, latest_display_event_id = display_engine.snapshot()

    return render_template(
        "main.html",
        status=stt_worker.status,
        is_listening=is_listening,
        initial_lines=initial_lines,
        latest_event_id=latest_event_id,
        latest_command=latest_command,
        initial_display_html=initial_display_html,
        latest_display_event_id=latest_display_event_id,
    )


@app.get("/transcript/events")
def stream_transcript() -> Response:
    last_event_id = (
        request.args.get("cursor", "").strip()
        or request.headers.get("Last-Event-ID", "").strip()
    )
    try:
        cursor = int(last_event_id) if last_event_id else 0
    except ValueError:
        cursor = 0

    @stream_with_context
    def event_stream():
        nonlocal cursor
        while True:
            cursor, pending = transcripts.wait_for_pending(cursor=cursor, timeout=15.0)
            if not pending:
                yield ": keep-alive\n\n"
                continue

            for event_id, line in pending:
                payload = json.dumps({"line": line})
                yield f"id: {event_id}\ndata: {payload}\n\n"
                cursor = event_id

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/display/events")
def stream_display() -> Response:
    last_event_id = (
        request.args.get("cursor", "").strip()
        or request.headers.get("Last-Event-ID", "").strip()
    )
    try:
        cursor = int(last_event_id) if last_event_id else 0
    except ValueError:
        cursor = 0

    @stream_with_context
    def event_stream():
        nonlocal cursor
        while True:
            cursor, pending = display_engine.wait_for_pending(cursor=cursor, timeout=15.0)
            if not pending:
                yield ": keep-alive\n\n"
                continue

            for event_id, payload in pending:
                yield f"id: {event_id}\ndata: {json.dumps(payload)}\n\n"
                cursor = event_id

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/toggle")
def toggle_listening():
    if stt_worker.status == "listening":
        stt_worker.pause_listening()
    else:
        stt_worker.start_listening()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)
