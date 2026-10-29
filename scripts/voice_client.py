"""Interactive reference client for the /voice WebSocket.

A faithful implementation of the wire protocol, useful for manual testing and
as a reference for the Angular client.

    python scripts/voice_client.py                      # microphone
    python scripts/voice_client.py --text               # keyboard only
    python scripts/voice_client.py --url ws://host/...  # another deployment

Requires ``sounddevice`` for microphone mode (dev dependency).
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
import wave

DEFAULT_URL = "ws://127.0.0.1:8000/api/v1/voice"
SAMPLE_RATE = 16_000
RECORD_SECONDS = 4.0


def record_wav(seconds: float = RECORD_SECONDS) -> bytes:
    import sounddevice as sd

    print(f"\n[mic] speak now ({seconds:.0f}s)...", flush=True)
    frames = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(frames.tobytes())

    print("[mic] captured", flush=True)
    return buffer.getvalue()


def play_wav(payload: bytes) -> None:
    import numpy as np
    import sounddevice as sd

    with wave.open(io.BytesIO(payload), "rb") as handle:
        channels = handle.getnchannels()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    audio = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        audio = audio.reshape(-1, channels)

    sd.play(audio, rate)
    sd.wait()


async def prompt_text() -> str:
    return (await asyncio.to_thread(input, "you> ")).strip()


async def run(url: str, *, use_text: bool, mute: bool) -> None:
    import websockets

    async with websockets.connect(
        url,
        open_timeout=60,
        max_size=16 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=60,
    ) as socket:
        print(f"connected to {url}")

        pending_header: dict | None = None
        state = "connecting"

        async def send_turn() -> None:
            if use_text:
                text = await prompt_text()
                if text:
                    await socket.send(json.dumps({"type": "text", "text": text}))
            else:
                await socket.send(await asyncio.to_thread(record_wav))

        async for message in socket:
            if isinstance(message, (bytes, bytearray)):
                label = (pending_header or {}).get("prompt") or "agent"
                print(f"[audio] {label} ({len(message)} bytes)")

                if not mute:
                    try:
                        await asyncio.to_thread(play_wav, bytes(message))
                    except Exception as exc:  # playback is best-effort
                        print(f"[audio] playback failed: {exc}")

                pending_header = None
                continue

            event = json.loads(message)
            kind = event.get("type")

            if kind in ("prompt.audio", "agent.audio"):
                pending_header = event
                continue

            print(f"<< {json.dumps(event, ensure_ascii=False)}")

            if kind == "state.changed":
                state = event["state"]
                if state in ("language_selection", "active"):
                    await send_turn()

            elif kind == "language.rejected":
                await send_turn()

            elif kind in ("language.failed", "session.closed"):
                break

            elif kind == "error" and event.get("fatal"):
                break


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--text", action="store_true", help="type instead of using the microphone"
    )
    parser.add_argument("--mute", action="store_true", help="do not play prompt audio")
    args = parser.parse_args()

    try:
        asyncio.run(run(args.url, use_text=args.text, mute=args.mute))
    except KeyboardInterrupt:
        print("\ninterrupted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
