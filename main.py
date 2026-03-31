"""
Convolution reverb tool — Python port of the C implementation.
Usage: python convolve.py <input.wav> <impulse.wav> <output.wav> [-m naive|parallel]

Dependencies:
    pip install numpy scipy soundfile
"""

import argparse
import sys
import time
import numpy as np
import soundfile as sf
import math
from scipy.fft import next_fast_len, rfft, irfft


# ---------------------------------------------------------------------------
# Naive — direct O(N*M) convolution, mirrors the C loop exactly
# ---------------------------------------------------------------------------


def convolve_naive(
    x: np.ndarray,  # (N, channels) float32
    h: np.ndarray,  # (M,) or (M, channels) float32
) -> np.ndarray:
    N, in_ch = x.shape
    M = h.shape[0]
    h_ch = 1 if h.ndim == 1 else h.shape[1]
    out_size = N + M - 1
    y = np.zeros((out_size, in_ch), dtype=np.float32)

    for n in range(out_size):
        k_start = (n - N + 1) if n >= N else 0
        k_end = n if n < M else (M - 1)

        # Vectorised over channels, same arithmetic as the C macros
        k_idx = np.arange(k_start, k_end + 1)  # shape (K,)
        x_idx = n - k_idx  # X[n-k, :]

        if h_ch == 1:
            hk = h[k_idx]  # (K,)
            y[n] = x[x_idx].T @ hk  # (ch,)
        else:
            hk = h[k_idx]  # (K, ch)
            y[n] = (x[x_idx] * hk).sum(axis=0)  # (ch,)

        if n % 2048 == 0:
            update_progress_bar(n, out_size)

    update_progress_bar(out_size, out_size)
    return y


# ---------------------------------------------------------------------------
# FFT-based — overlap-add convolution (O(N log M), optimal for long IRs)
# ---------------------------------------------------------------------------


def choose_block_size(M):
    if M < 2048:
        return 4096
    elif M < 16384:
        return 8192
    else:
        return 16384


def convolve_fft(
    x: np.ndarray,  # (N, channels)
    h: np.ndarray,  # (M,) or (M, channels)
) -> np.ndarray:
    N, in_ch = x.shape
    M = h.shape[0]
    h_ch = 1 if h.ndim == 1 else h.shape[1]
    out_size = N + M - 1

    # --- Tunable block size (IMPORTANT) ---
    L = choose_block_size(M)
    fft_size = next_fast_len(L + M - 1)

    # --- Prepare impulse FFT ---
    if h_ch == 1:
        H = rfft(h, n=fft_size)[:, None]  # shape (freq, 1)
        H = np.repeat(H, in_ch, axis=1)  # broadcast to channels
    else:
        H = rfft(h, n=fft_size, axis=0)  # (freq, channels)

    # --- Output buffer ---
    y = np.zeros((out_size, in_ch), dtype=np.float64)

    total_blocks = math.ceil(N / L)

    # --- Preallocate block buffer (avoid realloc) ---
    block_buf = np.zeros((L, in_ch), dtype=np.float32)

    for block_idx, start in enumerate(range(0, N, L)):
        end = min(start + L, N)
        block_len = end - start

        # Copy into buffer (zero-padded)
        block_buf[:block_len] = x[start:end]
        if block_len < L:
            block_buf[block_len:] = 0

        # --- FFT (all channels at once) ---
        X = rfft(block_buf, n=fft_size, axis=0)

        # --- Multiply in frequency domain ---
        Y = X * H

        # --- IFFT ---
        y_block = irfft(Y, n=fft_size, axis=0)

        # --- Overlap-add ---
        seg_len = min(fft_size, out_size - start)
        y[start : start + seg_len] += y_block[:seg_len]

        update_progress_bar(block_idx + 1, total_blocks)

    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # --- Load input ---
    try:
        x, sr_x = sf.read(args.input, dtype="float32", always_2d=True)
    except Exception as e:
        print(f"Error: Could not open input file '{args.input}': {e}", file=sys.stderr)
        sys.exit(1)

    # --- Load impulse ---
    try:
        h, sr_h = sf.read(args.impulse, dtype="float32", always_2d=True)
    except Exception as e:
        print(
            f"Error: Could not open impulse file '{args.impulse}': {e}", file=sys.stderr
        )
        sys.exit(1)

    # --- Validate ---
    in_ch = x.shape[1]
    imp_ch = h.shape[1]

    if imp_ch != 1 and imp_ch != in_ch:
        print("Error: Impulse must be mono or match input channels", file=sys.stderr)
        sys.exit(1)

    if sr_x != sr_h:
        print(
            f"Error: Sample rates do not match! Input: {sr_x}, Impulse: {sr_h}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Squeeze mono impulse to 1-D to match C macro logic (H(k) vs H(k,ch))
    if imp_ch == 1:
        h = h[:, 0]

    N, M = x.shape[0], h.shape[0]
    mode_label = {
        "naive": "Naive O(N×M)",
        "fft": "FFT overlap-add O(N log M)",
    }[args.mode]

    print(f"Mode: {mode_label}")
    print(
        f"Processing: ({N} samples, {in_ch} channels) * ({M} samples, {imp_ch} channels)"
    )

    # --- Convolve ---
    t0 = time.perf_counter()
    if args.mode == "fft":
        y = convolve_fft(x, h)
    else:
        y = convolve_naive(x, h)
    elapsed = time.perf_counter() - t0

    print(f"Time taken: {elapsed:.4f} seconds")

    # --- Write output ---
    try:
        sf.write(args.output, y, sr_x, subtype="FLOAT")
    except Exception as e:
        print(
            f"Error: Could not write output file '{args.output}': {e}", file=sys.stderr
        )
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convolution reverb — Python port",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Input WAV file")
    parser.add_argument("impulse", help="Impulse response WAV file")
    parser.add_argument("output", help="Output WAV file")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["naive", "fft"],
        default="fft",
        help="Convolution mode: fft (default, overlap-add) or naive (O(N*M), slow)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def update_progress_bar(current: int, total: int, bar_width: int = 40) -> None:
    if total == 0:
        return
    progress = current / total
    pos = int(bar_width * progress)
    bar = "=" * pos + (">" if pos < bar_width else "") + " " * (bar_width - pos - 1)
    print(f"\r[{bar}] {int(progress * 100)}%", end="", flush=True)
    if current >= total:
        print()  # newline on completion


if __name__ == "__main__":
    main()
