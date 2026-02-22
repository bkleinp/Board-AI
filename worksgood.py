#!/usr/bin/env python3
"""Live microphone transcription using faster-whisper."""

from __future__ import annotations

import argparse
import queue
import sys
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Listen to microphone and print speech-to-text continuously."
    )
    parser.add_argument(
        "--model",
        default="base.en",
        help="faster-whisper model size/name (default: base.en)",
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
        default=3.0,
        help="Audio chunk duration for each transcription call (default: 3.0)",
    )
    parser.add_argument(
        "--mic-device",
        default=None,
        help="Optional microphone device name/index for sounddevice",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    def audio_callback(indata: np.ndarray, frames: int, callback_time, status) -> None:
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        audio_queue.put(indata.copy())

    print(
        f"Loading model '{args.model}' on {args.device} ({args.compute_type})...",
        flush=True,
    )
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
    )

    print("Listening... Press Ctrl+C to stop.", flush=True)
    chunk_samples = int(args.sample_rate * args.chunk_seconds)
    buffer = np.zeros((0,), dtype=np.float32)

    try:
        with sd.InputStream(
            device=args.mic_device,
            channels=1,
            samplerate=args.sample_rate,
            dtype="float32",
            callback=audio_callback,
        ):
            while True:
                try:
                    frame = audio_queue.get(timeout=0.25)
                    buffer = np.concatenate((buffer, frame[:, 0]))
                except queue.Empty:
                    pass

                if buffer.shape[0] < chunk_samples:
                    continue

                audio_chunk = buffer[:chunk_samples]
                buffer = buffer[chunk_samples:]

                segments, _ = model.transcribe(
                    audio_chunk,
                    language=args.language,
                    vad_filter=True,
                    beam_size=1,
                )

                text = " ".join(segment.text.strip() for segment in segments).strip()
                if text:
                    print(text, flush=True)
                else:
                    print("[silence]", flush=True)

    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
        return 0
    except Exception as exc:  # pragma: no cover - runtime guard
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

