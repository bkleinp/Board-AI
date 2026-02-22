#!/usr/bin/env python3
"""Live microphone transcription using faster-whisper."""

from __future__ import annotations

import argparse
import difflib
import queue
import re
import sys
import threading
import time
from typing import Callable

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


WAKE_WORD = "orange"
WAKE_WORD_VARIANTS = (
    WAKE_WORD,
    "ornge",
    "orenge",
    "orinj",
    "oranje",
    "board",
    "bored",
)

STOP_WORD = "please"
STOP_WORD_VARIANTS = (
    STOP_WORD,
    "pleese",
    "pls",
    "plz",
    "pleasee",
)

FUZZY_MATCH_THRESHOLD = 0.78
AUDIO_QUEUE_MAX_FRAMES = 256
TRANSCRIPT_QUEUE_MAX_LINES = 1024
OUTPUT_QUEUE_MAX_LINES = 2048
TEXT_QUEUE_MAX_ITEMS = 1024
DEFAULT_TRANSCRIBE_MODEL = "base.en"
DEFAULT_CHUNK_SECONDS = 3.0
DEFAULT_BEAM_SIZE = 1


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
    return [token for token in cleaned.split() if token]


class SpeechToTextWorker:
    """Background speech-to-text worker with start/pause controls."""

    def __init__(
        self,
        model_name: str = DEFAULT_TRANSCRIBE_MODEL,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
        sample_rate: int = 16000,
        chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
        mic_device: str | int | None = None,
        transcript_callback: Callable[[str], None] | None = None,
        beam_size: int = DEFAULT_BEAM_SIZE,
        temperature: float = 0.0,
        vad_filter: bool = True,
        condition_on_previous_text: bool = False,
    ) -> None:
        self.model_name = model_name
        self.model_fallback_name = "base.en"
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.mic_device = mic_device
        self.transcript_callback = transcript_callback
        self.beam_size = beam_size
        self.temperature = temperature
        self.vad_filter = vad_filter
        self.condition_on_previous_text = condition_on_previous_text

        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=AUDIO_QUEUE_MAX_FRAMES)
        self.text_queue: queue.Queue[str] = queue.Queue(maxsize=TEXT_QUEUE_MAX_ITEMS)
        self.transcript_queue: queue.Queue[str] = queue.Queue(maxsize=TRANSCRIPT_QUEUE_MAX_LINES)
        self.output_queue: queue.Queue[str] = queue.Queue(maxsize=OUTPUT_QUEUE_MAX_LINES)
        self.listen_event = threading.Event()
        self.shutdown_event = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self.parser_thread: threading.Thread | None = None
        self.dispatch_thread: threading.Thread | None = None
        self.output_thread: threading.Thread | None = None
        self.model: WhisperModel | None = None
        self.active_model_name: str | None = None
        self.model_lock = threading.Lock()
        self.command_lock = threading.RLock()
        self.command_active = False
        self.command_tokens: list[str] = []
        self.wake_variants: list[str] = []
        for variant in WAKE_WORD_VARIANTS:
            tokens = _tokenize(variant)
            if tokens:
                self.wake_variants.append(tokens[0])
        self.stop_variants = []
        for variant in STOP_WORD_VARIANTS:
            tokens = _tokenize(variant)
            if tokens:
                self.stop_variants.append(tuple(tokens))

    def ensure_worker_started(self) -> None:
        if self.output_thread is None or not self.output_thread.is_alive():
            self.output_thread = threading.Thread(target=self._emit_output_lines, daemon=True)
            self.output_thread.start()
        if self.parser_thread is None or not self.parser_thread.is_alive():
            self.parser_thread = threading.Thread(target=self._parse_transcript_text, daemon=True)
            self.parser_thread.start()
        if self.transcript_callback is not None and (
            self.dispatch_thread is None or not self.dispatch_thread.is_alive()
        ):
            self.dispatch_thread = threading.Thread(target=self._dispatch_transcripts, daemon=True)
            self.dispatch_thread.start()
        if self.worker_thread and self.worker_thread.is_alive():
            return
        self.worker_thread = threading.Thread(target=self._run, daemon=True)
        self.worker_thread.start()

    def start_listening(self) -> None:
        self.ensure_worker_started()
        self.listen_event.set()

    def pause_listening(self) -> None:
        command_line = self._consume_active_command_line()
        if command_line:
            self._queue_output(command_line)
        self.listen_event.clear()

    def shutdown(self) -> None:
        command_line = self._consume_active_command_line()
        if command_line:
            self._queue_output(command_line)
        self.listen_event.clear()
        self.shutdown_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        if self.parser_thread and self.parser_thread.is_alive():
            self.parser_thread.join(timeout=1.0)
        if self.output_thread and self.output_thread.is_alive():
            self.output_thread.join(timeout=1.0)
        if self.dispatch_thread and self.dispatch_thread.is_alive():
            self.dispatch_thread.join(timeout=1.0)

    @property
    def status(self) -> str:
        return "listening" if self.listen_event.is_set() else "paused"

    def _audio_callback(self, indata: np.ndarray, frames: int, callback_time, status) -> None:
        del frames, callback_time
        if status:
            print(f"[audio] {status}", file=sys.stderr, flush=True)
        if self.listen_event.is_set():
            self._queue_latest_audio(indata.copy())

    def _queue_latest_audio(self, frame: np.ndarray) -> None:
        try:
            self.audio_queue.put_nowait(frame)
        except queue.Full:
            # Never block the realtime callback; keep the newest frame.
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.audio_queue.put_nowait(frame)
            except queue.Full:
                pass

    def _load_model(self) -> WhisperModel:
        if self.model is None:
            with self.model_lock:
                if self.model is None:
                    candidates = [self.model_name]
                    if self.model_name != self.model_fallback_name:
                        candidates.append(self.model_fallback_name)

                    last_error: Exception | None = None
                    for candidate in candidates:
                        try:
                            print(
                                f"Loading model '{candidate}' on {self.device} ({self.compute_type})...",
                                flush=True,
                            )
                            self.model = WhisperModel(
                                candidate,
                                device=self.device,
                                compute_type=self.compute_type,
                            )
                            self.active_model_name = candidate
                            if candidate != self.model_name:
                                print(
                                    f"[stt] Falling back to model '{candidate}'.",
                                    file=sys.stderr,
                                    flush=True,
                                )
                            break
                        except Exception as exc:  # pragma: no cover - runtime guard
                            last_error = exc
                            print(
                                f"[stt] Failed to load model '{candidate}': {exc}",
                                file=sys.stderr,
                                flush=True,
                            )

                    if self.model is None and last_error is not None:
                        raise last_error
        return self.model

    def _token_matches(self, heard: str, target: str) -> bool:
        if heard == target:
            return True
        return difflib.SequenceMatcher(None, heard, target).ratio() >= FUZZY_MATCH_THRESHOLD

    def _find_wake_index(self, tokens: list[str]) -> int | None:
        for idx, token in enumerate(tokens):
            if any(self._token_matches(token, wake) for wake in self.wake_variants):
                return idx + 1
        return None

    def _find_stop_index(self, tokens: list[str]) -> int | None:
        for idx in range(len(tokens)):
            for stop_phrase in self.stop_variants:
                phrase_len = len(stop_phrase)
                if idx + phrase_len > len(tokens):
                    continue
                window = tokens[idx : idx + phrase_len]
                if all(self._token_matches(window[i], stop_phrase[i]) for i in range(phrase_len)):
                    return idx
        return None

    def _consume_active_command_line(self) -> str | None:
        with self.command_lock:
            if not self.command_active:
                return None
            command = " ".join(self.command_tokens).strip()
            self.command_tokens = []
            self.command_active = False
            if command:
                return f'COMMAND FOUND: "{command}"'
        return None

    def _queue_output(self, line: str) -> None:
        try:
            self.output_queue.put_nowait(line)
        except queue.Full:
            # Keep UI responsive by dropping oldest output when overwhelmed.
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.output_queue.put_nowait(line)
            except queue.Full:
                pass

    def _notify_transcript(self, line: str) -> None:
        if self.transcript_callback is None:
            return
        try:
            self.transcript_queue.put_nowait(line)
        except queue.Full:
            # Keep queue fresh under pressure by dropping the oldest line.
            try:
                self.transcript_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.transcript_queue.put_nowait(line)
            except queue.Full:
                pass

    def _queue_text(self, text: str) -> None:
        try:
            self.text_queue.put_nowait(text)
        except queue.Full:
            # Keep most recent text if parser thread lags briefly.
            try:
                self.text_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.text_queue.put_nowait(text)
            except queue.Full:
                pass

    def _parse_transcript_text(self) -> None:
        while not self.shutdown_event.is_set() or not self.text_queue.empty():
            try:
                text = self.text_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._process_transcript_chunk(text)

    def _emit_output_lines(self) -> None:
        while not self.shutdown_event.is_set() or not self.output_queue.empty():
            try:
                line = self.output_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if not line.startswith('COMMAND FOUND: "'):
                print(line, flush=True)
            self._notify_transcript(line)

    def _dispatch_transcripts(self) -> None:
        if self.transcript_callback is None:
            return
        while not self.shutdown_event.is_set() or not self.transcript_queue.empty():
            try:
                line = self.transcript_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.transcript_callback(line)
            except Exception as exc:  # pragma: no cover - runtime guard
                print(f"[callback] {exc}", file=sys.stderr, flush=True)

    def _drain_audio_queue(self) -> None:
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def _process_transcript_chunk(self, text: str) -> None:
        lines_to_emit: list[str] = []
        with self.command_lock:
            if text:
                lines_to_emit.append(text)
                tokens = _tokenize(text)
                if not tokens:
                    command_line = self._consume_active_command_line()
                    if command_line:
                        lines_to_emit.append(command_line)
                    for line in lines_to_emit:
                        self._queue_output(line)
                    return

                if not self.command_active:
                    wake_index = self._find_wake_index(tokens)
                    if wake_index is None:
                        for line in lines_to_emit:
                            self._queue_output(line)
                        return
                    self.command_active = True
                    tokens = tokens[wake_index:]

                if not tokens:
                    for line in lines_to_emit:
                        self._queue_output(line)
                    return

                stop_index = self._find_stop_index(tokens)
                if stop_index is None:
                    self.command_tokens.extend(tokens)
                else:
                    self.command_tokens.extend(tokens[:stop_index])
                    command_line = self._consume_active_command_line()
                    if command_line:
                        lines_to_emit.append(command_line)
            else:
                lines_to_emit.append("[silence]")
                command_line = self._consume_active_command_line()
                if command_line:
                    lines_to_emit.append(command_line)

        for line in lines_to_emit:
            self._queue_output(line)

    def _run(self) -> None:
        chunk_samples = int(self.sample_rate * self.chunk_seconds)
        frame_parts: list[np.ndarray] = []
        buffered_samples = 0

        while not self.shutdown_event.is_set():
            try:
                model = self._load_model()
            except Exception as exc:  # pragma: no cover - runtime guard
                print(f"Error loading model: {exc}", file=sys.stderr, flush=True)
                time.sleep(2.0)
                continue

            if not self.listen_event.is_set():
                frame_parts = []
                buffered_samples = 0
                self._drain_audio_queue()
                time.sleep(0.1)
                continue

            try:
                with sd.InputStream(
                    device=self.mic_device,
                    channels=1,
                    samplerate=self.sample_rate,
                    dtype="float32",
                    callback=self._audio_callback,
                ):
                    print("[stt] Listening started.", flush=True)
                    while self.listen_event.is_set() and not self.shutdown_event.is_set():
                        try:
                            frame = self.audio_queue.get(timeout=0.25)
                            mono = frame[:, 0]
                            frame_parts.append(mono)
                            buffered_samples += mono.shape[0]
                        except queue.Empty:
                            continue

                        if buffered_samples < chunk_samples:
                            continue

                        merged = (
                            np.concatenate(frame_parts)
                            if len(frame_parts) > 1
                            else frame_parts[0]
                        )
                        offset = 0

                        while merged.shape[0] - offset >= chunk_samples:
                            chunk = merged[offset : offset + chunk_samples]
                            offset += chunk_samples

                            segments, _ = model.transcribe(
                                chunk,
                                language=self.language,
                                vad_filter=self.vad_filter,
                                beam_size=self.beam_size,
                                temperature=self.temperature,
                                without_timestamps=True,
                                condition_on_previous_text=self.condition_on_previous_text,
                            )
                            text = " ".join(seg.text.strip() for seg in segments).strip()
                            self._queue_text(text)

                        remainder = merged[offset:]
                        if remainder.shape[0] > 0:
                            frame_parts = [remainder]
                            buffered_samples = remainder.shape[0]
                        else:
                            frame_parts = []
                            buffered_samples = 0

                    print("[stt] Listening paused.", flush=True)
            except Exception as exc:  # pragma: no cover - runtime guard
                print(f"Error: {exc}", file=sys.stderr, flush=True)
                time.sleep(1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Listen to microphone and print speech-to-text continuously."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_TRANSCRIBE_MODEL,
        help=f"faster-whisper model size/name (default: {DEFAULT_TRANSCRIBE_MODEL})",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Whisper compute device: cpu, cuda, auto (default: cpu)",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="Whisper compute type, e.g. int8/float16/float32 (default: int8)",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code, e.g. en/es/fr (default: en)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Microphone sample rate (default: 16000)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=DEFAULT_CHUNK_SECONDS,
        help=f"Audio chunk duration for each transcription call (default: {DEFAULT_CHUNK_SECONDS})",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=DEFAULT_BEAM_SIZE,
        help=f"Whisper beam size (default: {DEFAULT_BEAM_SIZE})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Whisper decoding temperature (default: 0.0)",
    )
    parser.add_argument(
        "--no-vad-filter",
        action="store_true",
        help="Disable VAD filtering (enabled by default).",
    )
    parser.add_argument(
        "--condition-on-previous-text",
        action="store_true",
        help="Enable conditioning on previous text (disabled by default).",
    )
    parser.add_argument(
        "--mic-device",
        default=None,
        help="Optional microphone device name/index for sounddevice",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    worker = SpeechToTextWorker(
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        sample_rate=args.sample_rate,
        chunk_seconds=args.chunk_seconds,
        mic_device=args.mic_device,
        beam_size=args.beam_size,
        temperature=args.temperature,
        vad_filter=not args.no_vad_filter,
        condition_on_previous_text=args.condition_on_previous_text,
    )

    print("Listening... Press Ctrl+C to stop.", flush=True)
    worker.start_listening()

    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
        return 0
    finally:
        worker.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
