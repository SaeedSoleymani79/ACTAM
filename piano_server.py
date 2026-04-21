"""
piano_server.py  —  ACTAM Pro Audio Engine v2
═══════════════════════════════════════════════════════════

Instruments:
  • Grand Piano   — Steinway-inspired additive synthesis
                    (inharmonicity, hammer noise, tri-chord chorus)
  • Flamenco Guitar — Karplus-Strong physical model
                    (nylon string damping, body warmth)

Effects Chain (fully numpy-vectorised, no Python loops):
  • Tremolo  — smooth sine-wave LFO, continuous phase
  • Delay    — tape-echo circular buffer, high-freq rolloff
  • Reverb   — 8-comb + 4-allpass Freeverb/Schroeder model

WebSocket protocol (browser → server):
  { "type": "note_on",  "id": <int>,   "freq": <float> }
  { "type": "note_off", "id": <int>                    }
  { "type": "switch",   "vst": "piano"|"guitar"        }
  { "type": "param",    "name": <str>, "val": <any>    }

Install dependencies once:
  pip install websockets numpy sounddevice scipy
"""

import asyncio
import json
import sys
import threading

import numpy as np
from scipy.signal import lfilter, butter, sosfilt
import sounddevice as sd
import websockets

# ──────────────────────────────────────────────────────────────
# Global config
# ──────────────────────────────────────────────────────────────

SR         = 44100     # sample rate (Hz)
WS_HOST    = "localhost"
WS_PORT    = 8765
BLOCK      = 512       # audio callback block size  (~11 ms latency)
NOTE_DUR   = 6.0       # seconds of audio to pre-generate per note

# ──────────────────────────────────────────────────────────────
# Shared live state
# ──────────────────────────────────────────────────────────────

# Which virtual instrument is active
current_vst  = "piano"

# All sounding notes: id → { data, pos(float), on, rel_pos }
# 'pos' is a float so we can do fractional-index pitch shifting.
active_notes = {}
notes_lock   = threading.Lock()

# Real-time controllable parameters
params = {
    "tune":            0.0,    # fine-tune offset  (±2 semitones)
    "pitch_bend":      0.0,    # pitch wheel        (±2 semitones)
    "tremolo_on":      False,
    "tremolo_rate":    5.0,    # LFO rate (Hz)
    "tremolo_depth":   0.5,    # 0 = off, 1 = full AM
    "delay_on":        False,
    "delay_time":      0.33,   # echo delay (seconds)
    "delay_feedback":  0.45,   # how much of each echo bleeds into the next
    "delay_level":     0.50,   # wet/dry mix for delay
    "reverb_on":       False,
    "reverb_mix":      0.40,   # wet/dry mix for reverb
}
param_lock   = threading.Lock()

# Pre-computed note waveforms (MIDI int → np.float32 array)
piano_lib    = {}
guitar_lib   = {}

# ──────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────

def midi2freq(m: int) -> float:
    """Equal-temperament: A4 (MIDI 69) = 440 Hz."""
    return 440.0 * (2.0 ** ((m - 69) / 12.0))


def note_lib(midi_note: int) -> np.ndarray:
    """
    Fetch the pre-computed waveform for the active instrument.
    If the MIDI note is out of our generated range, clamp to the nearest edge.
    """
    lib = piano_lib if current_vst == "piano" else guitar_lib
    m   = max(min(int(midi_note), max(lib)), min(lib))
    # Return a copy so playback position drift doesn't corrupt the master buffer.
    return lib[m]


# ──────────────────────────────────────────────────────────────
# ① Steinway-Inspired Grand Piano Synthesis
# ──────────────────────────────────────────────────────────────

def make_piano(midi_note: int, dur: float = NOTE_DUR) -> np.ndarray:
    """
    Builds a grand piano note via additive (Fourier) synthesis.

    What makes it sound like a Steinway rather than a buzz-box:

    ① Inharmonicity  — real piano strings are stiff wire, so harmonics
       drift slightly sharp of integer multiples.  Standard formula:
           f_n  ≈  n · f₀ · √(1 + B·n²)
       B (inharmonicity coefficient) is larger for thick bass strings,
       near zero for thin treble strings.

    ② Differential decay  — upper harmonics fade faster than the
       fundamental.  That's why a piano sounds bright on the attack
       and warm on the sustain.

    ③ Tri-chord chorus  — a real Steinway has 2–3 strings per note,
       tuned ±0.25 cents apart.  We add two slightly-detuned copies of
       the lower 4 harmonics to reproduce that gentle beating effect.

    ④ Hammer transient  — a <15 ms shaped noise burst at note onset,
       simulating felt meeting steel at the moment of impact.
    """
    n_samp = int(dur * SR)
    t      = np.arange(n_samp, dtype=np.float32) / SR
    f0     = midi2freq(midi_note)

    # Map MIDI 21→108 (A0→C8) to 0→1 for parameter interpolation
    norm = float(np.clip((midi_note - 21) / 87.0, 0, 1))

    # Bass strings are stiffer → higher B.  Treble strings are thin → tiny B.
    B    = 1.6e-4 * (1.0 - norm) + 4e-6 * norm

    # Higher notes produce more upper-partial energy (brighter timbre)
    brt  = 0.30 + 0.70 * norm

    # (harmonic index, amplitude, base_decay_rate_at_mid_register)
    partials = [
        (1,  1.000,  0.35),
        (2,  0.720,  0.88),
        (3,  0.510,  1.75),
        (4,  0.350 * brt, 3.10),
        (5,  0.240 * brt, 4.90),
        (6,  0.160 * brt, 7.20),
        (7,  0.100 * brt, 10.0),
        (8,  0.065 * brt, 13.5),
        (9,  0.040 * brt, 17.5),
        (10, 0.025 * brt, 22.5),
        (11, 0.015 * brt, 28.0),
        (12, 0.008 * brt, 35.0),
    ]

    sig = np.zeros(n_samp, dtype=np.float32)

    for h, amp, d0 in partials:
        # Inharmonic frequency for this partial
        fh    = f0 * h * float(np.sqrt(1.0 + B * h * h))
        # Treble notes decay faster globally
        decay = d0 * (0.75 + 0.50 * norm)
        sig  += amp * np.sin(2 * np.pi * fh * t) * np.exp(-decay * t)

    # Tri-chord chorus: two extra copies detuned by ±0.25 cents
    for det_cents in (-0.25, +0.25):
        ratio  = 2.0 ** (det_cents / 1200.0)
        chorus = np.zeros(n_samp, dtype=np.float32)
        for h, amp, d0 in partials[:4]:
            fh      = f0 * h * ratio
            chorus += (amp * 0.27) * np.sin(2*np.pi*fh*t) * np.exp(-d0*0.90*t)
        sig += chorus

    # Hammer knock: short noise burst with fast exponential decay
    an  = min(int(0.014 * SR), n_samp)
    rng = np.random.default_rng(int(midi_note))
    nz  = rng.standard_normal(an).astype(np.float32)
    # Soften the noise through a little low-pass (felt absorbs high-freq)
    nz  = np.convolve(nz, np.ones(6) / 6.0, mode='same').astype(np.float32)
    nz *= 0.18 * np.exp(-np.arange(an, dtype=np.float32) * 750 / SR)
    sig[:an] += nz

    # 1 ms linear ramp to kill the DC click at note onset
    fn       = min(int(0.001 * SR), n_samp)
    sig[:fn] *= np.linspace(0.0, 1.0, fn, dtype=np.float32)

    # Normalise: leave headroom for polyphonic mixing
    peak = float(np.max(np.abs(sig)))
    if peak > 0:
        sig *= 0.40 / peak

    return sig


# ──────────────────────────────────────────────────────────────
# ② Flamenco Guitar — Karplus-Strong Physical Model
# ──────────────────────────────────────────────────────────────

def make_guitar(midi_note: int, dur: float = NOTE_DUR) -> np.ndarray:
    """
    Nylon-string flamenco guitar using the Karplus-Strong algorithm.

    Physical intuition:
      Fill a ring-buffer of length N ≈ SR/f₀ with an excitation signal
      (the 'pluck').  Each output sample is the average of the two previous
      buffer values, scaled by a damping factor g:

          y[n] = g · ½ · (y[n-N] + y[n-N-1])

      The averaging is a one-pole low-pass filter, so each cycle strips a
      little high-frequency energy — exactly what a plucked string does.
      After a few hundred cycles the upper harmonics are gone and you're
      left with a clean, warm fundamental that decays slowly.

    We implement this as a IIR filter using scipy.signal.lfilter, which
    handles the large delay (hundreds of samples) efficiently without any
    Python loops.

    Nylon/flamenco character:
      • Bandpass-shaped excitation (200–3 500 Hz) → warm, not metallic.
      • Damping g ≈ 0.998 (nylon sustains longer than steel).
      • Post-processing low-pass blends in body warmth.
    """
    n_out = int(dur * SR)
    f0    = midi2freq(midi_note)

    # Number of samples that fit in one period at this pitch
    N     = max(2, int(round(SR / f0)))

    # Excitation noise — deterministic seed makes the same note reproducible
    rng   = np.random.default_rng(int(midi_note) * 7 + 13)
    exc   = rng.standard_normal(N).astype(np.float64)

    # Bandpass the excitation: nylon strings plucked near the soundhole
    # produce a warm bloom around 200–3 500 Hz, not a metallic ping.
    try:
        sos_bp = butter(2, [200 / (SR/2), 3500 / (SR/2)],
                        btype='band', output='sos')
        exc    = sosfilt(sos_bp, exc)
    except Exception:
        pass   # bandpass can fail for very low notes; raw noise is fine

    mx = np.max(np.abs(exc))
    if mx > 0:
        exc /= mx

    # Damping factor: higher notes (thinner strings) damp slightly faster
    norm    = float(np.clip((midi_note - 36) / 60.0, 0, 1))
    damping = 0.9986 - 0.0075 * norm

    # Build IIR denominator for the Karplus-Strong recurrence:
    #   a[0]*y[n] + a[N]*y[n-N] + a[N+1]*y[n-N-1] = b[0]*x[n]
    a_coef       = np.zeros(N + 2, dtype=np.float64)
    a_coef[0]    = 1.0
    a_coef[N]    = -damping * 0.5
    a_coef[N+1]  = -damping * 0.5

    x_buf        = np.zeros(n_out, dtype=np.float64)
    x_buf[:N]    = exc
    output       = lfilter([1.0], a_coef, x_buf).astype(np.float32)

    # Body warmth: mix a low-passed version in for the guitar cabinet colour
    try:
        sos_lp = butter(2, 7000 / (SR/2), btype='low', output='sos')
        warm   = sosfilt(sos_lp, output.astype(np.float64)).astype(np.float32)
        output = 0.72 * output + 0.28 * warm
    except Exception:
        pass

    # 1 ms anti-click fade-in
    fn            = min(int(0.001 * SR), n_out)
    output[:fn]  *= np.linspace(0.0, 1.0, fn, dtype=np.float32)

    peak = float(np.max(np.abs(output)))
    if peak > 0:
        output *= 0.40 / peak

    return output


# ──────────────────────────────────────────────────────────────
# Pre-computation  (done once at startup, then cached)
# ──────────────────────────────────────────────────────────────

MIDI_RANGE = range(36, 91)   # C2 (36) → Eb6 (87) – covers 3+ octaves on the keyboard


def precompute():
    """
    Synthesise every note for both instruments and store in the library dicts.
    We do this before accepting WebSocket connections so note_on is instant.
    """
    global piano_lib, guitar_lib

    print("🎹  Pre-computing Grand Piano waveforms…")
    for m in MIDI_RANGE:
        piano_lib[m] = make_piano(m)
        print(f"   Piano  MIDI {m:3d}  {midi2freq(m):8.2f} Hz  ✓", end='\r')
    print(f"\n   ✅  {len(piano_lib)} piano notes cached.\n")

    print("🎸  Pre-computing Flamenco Guitar waveforms (Karplus-Strong)…")
    for m in MIDI_RANGE:
        guitar_lib[m] = make_guitar(m)
        print(f"   Guitar MIDI {m:3d}  {midi2freq(m):8.2f} Hz  ✓", end='\r')
    print(f"\n   ✅  {len(guitar_lib)} guitar notes cached.\n")


# ──────────────────────────────────────────────────────────────
# DSP Effect state  (global so the audio callback can reach them)
# ──────────────────────────────────────────────────────────────

# ── Tape Delay ────────────────────────────────────────────────
DELAY_MAXN   = SR * 3               # 3 seconds of delay line storage
delay_buf    = np.zeros(DELAY_MAXN, dtype=np.float32)
delay_wptr   = 0                    # circular-buffer write pointer

# ── Freeverb / Schroeder Reverb ───────────────────────────────
# 8 parallel comb filter delays (in samples) tuned to avoid musical pitches.
# All are > 1 000 samples, which is larger than our block size (512),
# so the vectorised circular-buffer reads/writes never overlap within a block.
REV_COMB_D  = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116]
REV_COMB_FB = 0.84   # comb feedback (room decay speed)
rev_cbufs   = [np.zeros(d, dtype=np.float32) for d in REV_COMB_D]
rev_cptrs   = [0] * len(REV_COMB_D)

# 4 series allpass diffusers — these scatter the sound in time so the reverb
# doesn't sound like a cluster of discrete echoes.
# We use scipy lfilter with maintained zi (filter state) so the diffusion is
# continuous across audio callbacks.
AP_DELAYS   = [225, 556, 441, 341]
AP_G        = 0.5

def _build_allpass(delay: int, g: float = 0.5):
    """Build b/a coefficients for a Schroeder allpass filter of given delay."""
    b      = np.zeros(delay + 1, dtype=np.float64)
    b[0]   = -g
    b[-1]  = 1.0
    a      = np.zeros(delay + 1, dtype=np.float64)
    a[0]   = 1.0
    a[-1]  = -g
    return b, a

ap_filters  = [_build_allpass(d) for d in AP_DELAYS]
# zi holds the filter memory so phase is seamless between callbacks
ap_zi       = [np.zeros(d, dtype=np.float64) for d in AP_DELAYS]

# ── Tremolo LFO ───────────────────────────────────────────────
# Phase accumulates globally so the LFO never resets mid-note.
lfo_phase   = 0.0


# ──────────────────────────────────────────────────────────────
# Effect processors  (all vectorised — no Python loops)
# ──────────────────────────────────────────────────────────────

def fx_tremolo(sig: np.ndarray, rate: float, depth: float) -> np.ndarray:
    """
    Amplitude tremolo using a continuous sine-wave LFO.
    Depth=0 → no effect; Depth=1 → signal dips fully to zero at trough.
    """
    global lfo_phase
    n      = len(sig)
    t_rel  = lfo_phase + np.arange(n, dtype=np.float32) / SR
    lfo    = 1.0 - depth * (0.5 - 0.5 * np.cos(2.0 * np.pi * rate * t_rel))
    # Keep phase in a sensible range to avoid floating-point drift over long sessions
    lfo_phase = float(t_rel[-1]) % (1.0 / max(rate, 0.01))
    return (sig * lfo).astype(np.float32)


def fx_delay(sig: np.ndarray, delay_time: float,
             feedback: float, level: float) -> np.ndarray:
    """
    Tape-style echo with feedback.

    We read a block of delayed samples in one numpy indexing operation, then
    write the new block.  This is safe (no intra-block dependency) because
    the minimum delay time (100 ms ≈ 4 410 samples) is always much larger
    than our block size (512 samples).
    """
    global delay_buf, delay_wptr
    n         = len(sig)
    d_samp    = max(1, int(delay_time * SR))
    read_idx  = (delay_wptr - d_samp + np.arange(n, dtype=np.int64)) % DELAY_MAXN
    delayed   = delay_buf[read_idx]

    write_idx = (delay_wptr + np.arange(n, dtype=np.int64)) % DELAY_MAXN
    # Feedback path with a gentle high-cut to simulate tape head losses
    delay_buf[write_idx] = sig + feedback * delayed * 0.97
    delay_wptr           = int((delay_wptr + n) % DELAY_MAXN)

    out = sig + level * delayed
    np.clip(out, -1.5, 1.5, out=out)
    return out.astype(np.float32)


def fx_reverb(sig: np.ndarray, mix: float) -> np.ndarray:
    """
    Freeverb-inspired Schroeder reverb.

    Signal flows: input → 8 parallel comb filters (room resonances)
                       → sum → 4 series allpass filters (diffusion)
                       → wet/dry blend with original signal.

    The allpass filters use scipy.signal.lfilter with zi (filter state) so
    the decay tail is mathematically continuous between audio callbacks.
    """
    global rev_cbufs, rev_cptrs, ap_zi

    n   = len(sig)
    rev = np.zeros(n, dtype=np.float32)

    # Parallel comb filters — each models one room reflection path
    for i, dlen in enumerate(REV_COMB_D):
        idx          = (rev_cptrs[i] + np.arange(n, dtype=np.int64)) % dlen
        comb_out     = rev_cbufs[i][idx]
        rev_cbufs[i][idx] = sig + REV_COMB_FB * comb_out
        rev_cptrs[i]      = int((rev_cptrs[i] + n) % dlen)
        rev             += comb_out

    rev *= 0.125   # normalise across 8 parallel paths

    # Series allpass diffusers — smear the impulse so it sounds like a space
    for j, (b, a) in enumerate(ap_filters):
        y, ap_zi[j] = lfilter(b, a, rev.astype(np.float64), zi=ap_zi[j])
        rev          = y.astype(np.float32)

    return ((1.0 - mix) * sig + mix * rev).astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Audio callback
# ──────────────────────────────────────────────────────────────

def audio_callback(outdata: np.ndarray, frames: int, time_info, status) -> None:
    """
    sounddevice fires this every ~11 ms to fill the DAC buffer.

    Pitch shifting is done by reading the pre-computed waveform at a
    fractional step speed (speed = 2^(semitones/12)).  Pitch-up → read
    faster, pitch-down → read slower.  We use linear interpolation to
    reconstruct samples between integer positions.
    """
    mixed = np.zeros(frames, dtype=np.float32)

    with param_lock:
        p = params.copy()

    # Total pitch offset: fine-tune + pitch-wheel, converted to playback speed
    semitones = float(p['tune']) + float(p['pitch_bend'])
    speed     = float(2.0 ** (semitones / 12.0))

    with notes_lock:
        dead = []
        for nid, note in list(active_notes.items()):
            data    = note['data']
            pos     = float(note['pos'])
            dlen    = len(data)

            # If we've already read past the end of the waveform, retire the note
            if int(pos) >= dlen - 2:
                dead.append(nid)
                continue

            # Build a fractional index array for this block
            frac_idx = pos + np.arange(frames, dtype=np.float32) * speed
            int_idx  = frac_idx.astype(np.int64)
            np.clip(int_idx, 0, dlen - 2, out=int_idx)
            frac     = (frac_idx - int_idx).astype(np.float32)

            # Linear interpolation between adjacent samples
            chunk = data[int_idx] * (1.0 - frac) + data[int_idx + 1] * frac

            # Apply a fast release envelope when the key has been lifted
            if not note['on']:
                rel_t    = float(note['rel_pos']) + np.arange(frames, dtype=np.float32) / SR
                chunk   *= np.exp(-18.0 * rel_t)
                note['rel_pos'] = float(rel_t[-1])
                if note['rel_pos'] > 0.40:
                    dead.append(nid)

            mixed         += chunk
            note['pos']    = float(frac_idx[-1] + speed)  # advance past the last read sample
            if int(note['pos']) >= dlen - 2:
                dead.append(nid)

        for nid in dead:
            active_notes.pop(nid, None)

    # ── FX Chain ──────────────────────────────────────────────
    if p['tremolo_on']:
        mixed = fx_tremolo(mixed, float(p['tremolo_rate']), float(p['tremolo_depth']))
    if p['delay_on']:
        mixed = fx_delay(mixed, float(p['delay_time']),
                         float(p['delay_feedback']), float(p['delay_level']))
    if p['reverb_on']:
        mixed = fx_reverb(mixed, float(p['reverb_mix']))

    # Soft-clip limiter — tanh is gentle and musical, prevents clipping pops
    outdata[:, 0] = np.tanh(mixed * 0.85)


# ──────────────────────────────────────────────────────────────
# WebSocket handler
# ──────────────────────────────────────────────────────────────

async def ws_handler(ws) -> None:
    """
    Manages one browser connection.

    Messages arrive as JSON strings.  We parse them and update either the
    active_notes dict (for note events) or the params dict (for control changes).
    """
    global current_vst
    addr = getattr(ws, 'remote_address', '?')
    print(f"[ws] ↑ Connected:    {addr}")

    try:
        async for raw in ws:
            try:
                msg  = json.loads(raw)
                kind = msg.get('type')

                if kind == 'note_on':
                    # Convert the frequency the browser sends into the nearest MIDI note
                    nid  = msg['id']
                    freq = float(msg['freq'])
                    midi = round(12 * np.log2(max(freq, 8.0) / 440.0) + 69)
                    data = note_lib(midi)
                    with notes_lock:
                        active_notes[nid] = {
                            'data':    data,
                            'pos':     0.0,
                            'on':      True,
                            'rel_pos': 0.0,
                        }

                elif kind == 'note_off':
                    nid = msg['id']
                    with notes_lock:
                        if nid in active_notes:
                            active_notes[nid]['on']      = False
                            active_notes[nid]['rel_pos'] = 0.0

                elif kind == 'switch':
                    current_vst = msg.get('vst', 'piano')
                    print(f"[ws]   Instrument → {current_vst}")

                elif kind == 'param':
                    name = msg['name']
                    val  = msg['val']
                    with param_lock:
                        if name in params:
                            # Boolean toggles arrive as bool or int from JS
                            if isinstance(params[name], bool):
                                params[name] = bool(val)
                            else:
                                params[name] = float(val)

            except (json.JSONDecodeError, KeyError, TypeError):
                pass  # silently ignore malformed messages

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        print(f"[ws] ↓ Disconnected: {addr}")


# ──────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────

async def main() -> None:
    # 1. Synthesise all notes up front so note_on is instant.
    precompute()

    # 2. Start the real-time audio output stream.
    stream = sd.OutputStream(
        samplerate=SR,
        channels=1,
        dtype='float32',
        blocksize=BLOCK,
        callback=audio_callback,
    )
    stream.start()
    print(f"🔊  Audio engine live  (mono, {SR} Hz, block={BLOCK})\n")

    # 3. Accept WebSocket connections from the browser.
    async with websockets.serve(ws_handler, WS_HOST, WS_PORT):
        print(f"🌐  WebSocket server  →  ws://{WS_HOST}:{WS_PORT}")
        print("    Open  piano.html  in your browser and start playing!")
        print("    Ctrl-C to stop.\n")
        await asyncio.Future()   # run forever until interrupted


if __name__ == '__main__':
    # Windows requires the Selector policy for asyncio + subprocesses.
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('\n👋  Engine offline.  Goodbye!')
