"""
piano_server.py  –  Python WebSocket backend + audio engine for the Web Piano.

How it works:
  1. At startup we synthesize every piano note using additive synthesis
     (multiple sine-wave harmonics with different decay speeds – this is
     what makes it sound like a piano string instead of a boring beep).
  2. We open a sounddevice OutputStream whose audio callback mixes
     all currently-sounding notes together on every ~10 ms buffer fill.
     This gives us polyphony for free – press as many keys as you like.
  3. A WebSocket server listens for {"type":"keypress","key":"W"} messages
     from the browser and queues the matching pre-computed waveform.

Dependencies (install once):
  pip install websockets numpy sounddevice
"""

import asyncio
import json
import sys
import threading

import numpy as np
import sounddevice as sd
import websockets

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

SAMPLE_RATE    = 44100   # standard CD-quality sample rate
NOTE_DURATION  = 3.5     # how long each note ring (seconds)
OUTPUT_GAIN    = 0.35    # per-note volume – leaves headroom for chords

WS_HOST = "localhost"
WS_PORT = 8765

# Keyboard key  →  fundamental frequency (equal-temperament, A4 = 440 Hz)
# Covers one octave from middle-C (C4) to B4, with all five black notes.
NOTE_FREQUENCIES: dict[str, float] = {
    "W": 261.63,   # C4  – middle C
    "3": 277.18,   # C#4 / Db4
    "E": 293.66,   # D4
    "4": 311.13,   # D#4 / Eb4
    "R": 329.63,   # E4
    "T": 349.23,   # F4
    "6": 369.99,   # F#4 / Gb4
    "Y": 392.00,   # G4
    "7": 415.30,   # G#4 / Ab4
    "U": 440.00,   # A4  – concert A (tuning reference)
    "8": 466.16,   # A#4 / Bb4
    "I": 493.88,   # B4
}

# ──────────────────────────────────────────────────────────────
# Sound Synthesis  –  additive synthesis of piano harmonics
# ──────────────────────────────────────────────────────────────

def generate_piano_note(frequency: float, duration: float = NOTE_DURATION) -> np.ndarray:
    """
    Build a single piano note via additive synthesis.

    A real piano string vibrates at many frequencies simultaneously:
      • the fundamental (the note you hear)
      • an octave above (2×)
      • a fifth above that (3×)
      • and so on …

    Crucially, higher harmonics fade out faster than lower ones.
    We model that with per-harmonic exponential decay envelopes.
    The result sounds warm and bell-like – much more piano-ish than
    a raw sine wave.
    """
    t = np.linspace(0.0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)

    # (harmonic_number,  relative_amplitude,  decay_rate)
    # decay_rate is the exponent in  e^(-rate * t)  – larger = faster fade
    partials = [
        (1,  1.00,  1.2),   # fundamental  – rings longest
        (2,  0.55,  2.0),   # octave
        (3,  0.28,  3.8),   # perfect 12th
        (4,  0.14,  5.5),   # two octaves
        (5,  0.07,  8.0),   # major 17th
        (6,  0.04, 11.0),
        (7,  0.025, 14.0),
        (8,  0.015, 18.0),  # upper partial – adds brightness at attack
    ]

    signal = np.zeros(len(t), dtype=np.float32)
    for harmonic, amp, decay in partials:
        wave     = np.sin(2.0 * np.pi * frequency * harmonic * t)
        envelope = np.exp(-decay * t, dtype=np.float32)
        signal  += amp * wave * envelope

    # Short linear ramp at the very start (~8 ms) to suppress the
    # "click" that would otherwise happen when the waveform jumps from
    # silence to a non-zero value instantaneously.
    attack_n = int(0.008 * SAMPLE_RATE)
    signal[:attack_n] *= np.linspace(0.0, 1.0, attack_n, dtype=np.float32)

    # Normalise so the loudest point sits at OUTPUT_GAIN,
    # leaving plenty of headroom when several notes play at once.
    peak = np.max(np.abs(signal))
    if peak > 0.0:
        signal = signal * (OUTPUT_GAIN / peak)

    return signal


def precompute_all_notes() -> dict[str, np.ndarray]:
    """
    Synthesize every note once at startup and cache the results.
    This way pressing a key triggers playback instantly – no
    per-request synthesis lag.
    """
    print("🎹  Pre-computing piano waveforms …")
    cache: dict[str, np.ndarray] = {}
    for key, freq in NOTE_FREQUENCIES.items():
        cache[key] = generate_piano_note(freq)
        print(f"     {key:>2s}  →  {freq:7.2f} Hz  ✓")
    print(f"     All {len(cache)} notes ready.\n")
    return cache


# ──────────────────────────────────────────────────────────────
# Polyphonic Audio Engine
# ──────────────────────────────────────────────────────────────

# Every currently-sounding note is stored here as
#   {"data": np.ndarray, "pos": int}
# The audio callback drains these; play_note() appends to them.
active_notes: list[dict] = []
notes_lock   = threading.Lock()

# Filled by main() before the server starts accepting connections.
precomputed: dict[str, np.ndarray] = {}


def audio_callback(outdata: np.ndarray, frames: int, time_info, status) -> None:
    """
    sounddevice calls this function every ~10 ms to fill the speaker buffer.

    We walk every active note, copy the next `frames` samples from its
    waveform, and sum them all together.  Notes that have finished playing
    are removed from the list.  The summed signal goes straight to the
    speakers (or headphones).
    """
    if status:
        # Print any under/overflow warnings to stderr so they don't
        # clutter the normal console output.
        print(f"[audio] {status}", file=sys.stderr)

    mixed = np.zeros(frames, dtype=np.float32)

    with notes_lock:
        finished = []
        for note in active_notes:
            remaining = len(note["data"]) - note["pos"]
            if remaining <= 0:
                finished.append(note)
                continue
            n = min(frames, remaining)
            mixed[:n] += note["data"][note["pos"]: note["pos"] + n]
            note["pos"] += n
            if note["pos"] >= len(note["data"]):
                finished.append(note)
        for note in finished:
            active_notes.remove(note)

    # Hard-clip as a last-resort safety net (shouldn't trigger
    # in normal use because per-note gain is low, but belt-and-suspenders
    # is good practice for anything going to a speaker).
    np.clip(mixed, -1.0, 1.0, out=mixed)

    # sounddevice buffer is (frames × channels); we're mono so channel 0.
    outdata[:, 0] = mixed


def play_note(key: str) -> None:
    """
    Queue a pre-computed note for immediate playback.

    We just append to the list that the audio thread is already
    draining, so there's essentially zero scheduling delay.
    """
    key = key.upper()
    if key in precomputed:
        with notes_lock:
            active_notes.append({"data": precomputed[key], "pos": 0})


# ──────────────────────────────────────────────────────────────
# WebSocket Server
# ──────────────────────────────────────────────────────────────

async def handle_client(websocket) -> None:
    """
    Manage one browser connection.

    Expected message format (JSON):
        {"type": "keypress", "key": "W"}

    The server doesn't send anything back – it only listens and plays.
    """
    addr = getattr(websocket, "remote_address", "unknown")
    print(f"[ws]  ↑ Client connected    {addr}")
    try:
        async for raw_message in websocket:
            try:
                msg = json.loads(raw_message)
                if msg.get("type") == "keypress":
                    key = str(msg.get("key", "")).upper()
                    play_note(key)
                    # Uncomment the line below to log every keypress:
                    # print(f"[ws]  ♩  key={key}")
            except (json.JSONDecodeError, KeyError):
                # Ignore messages that don't match our protocol.
                pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        print(f"[ws]  ↓ Client disconnected {addr}")


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────

async def main() -> None:
    global precomputed

    # 1. Synthesize all notes before accepting any connections.
    precomputed = precompute_all_notes()

    # 2. Open the audio output stream.
    #    blocksize=512 gives ~11 ms of latency on most hardware –
    #    plenty responsive for musical performance.
    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=512,
        callback=audio_callback,
    )
    stream.start()
    print("🔊  Audio engine running  (mono, 44 100 Hz, block=512)\n")

    # 3. Start the WebSocket server and wait forever.
    async with websockets.serve(handle_client, WS_HOST, WS_PORT):
        print(f"🌐  WebSocket server →  ws://{WS_HOST}:{WS_PORT}")
        print("    Open  piano.html  in your browser and start playing!\n")
        print("    Press  Ctrl-C  to stop.\n")
        await asyncio.Future()   # keep running until Ctrl-C


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋  Server stopped.  Goodbye!")
