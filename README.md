# WAV Convolution Engine

A C-based convolution tool for WAV audio files, with naive, OpenMP, AVX2, and FFT-based implementations.

## Project Overview

This project implements a simple convolution engine that applies an Impulse Response (IR) to an input audio signal. It's primarily designed for educational purposes and testing basic audio DSP concepts.

- **Main Technologies:** C, [dr_wav](https://github.com/mackron/dr_libs) for audio I/O, [PFFFT](https://bitbucket.org/jpommier/pffft/) for frequency-domain convolution, and [nob.h](https://github.com/tsoding/nob.h) for the build system.
- **Key Features:**
  - Support for mono and stereo WAV files (interleaved).
  - Impulse responses can be mono or match the input channel count.
  - Naive, **Parallel (OpenMP)**, **SIMD (AVX2)**, and **Fast Fourier Transform (FFT / PFFFT)** convolution implementations.
  - High-resolution timing for performance benchmarking.
  - Real-time progress bar with throttled updates for the naive implementation.
  - Writes 32-bit float WAV output.
  - Automatic build system using `nob.c`, with compiler selection and debug/profiling flags.
  - Memory leak detection via `stb_leakcheck.h` (optional).

## Performance Optimizations

Several optimization techniques have been applied to the convolution engine:

- **Parallelization (OpenMP):** Utilizes multi-core processing to distribute the workload, achieving a ~5x speedup on typical hardware.
- **SIMD Vectorization (AVX2 + FMA):** Uses 256-bit wide registers and Fused Multiply-Add instructions to process 8 samples at once, providing massive throughput for time-domain convolution.
- **FFT Acceleration (PFFFT):** Accelerates convolution by performing it in the frequency domain. This reduces computational complexity from $O(N \cdot M)$ to $O(N \log N)$, providing massive speedups for long signals.
- **Interleaved Cache Locality:** Loops process all channels in a single pass. Since WAV files are interleaved (L, R, L, R...), this ensures sequential memory access, significantly reducing cache misses.
- **Branchless Inner Loop:** Loop bounds are pre-calculated for every output sample, removing conditional checks from the critical path.
- **Macro-based Sampling:** Uses zero-cost pre-processor macros (e.g., `X(n, c)`) for "math-like" readability of interleaved samples.

## Project Structure

- `main.c`: Core logic for loading WAV files, convolution engines, and performance timing.
- `include/`: Third-party libraries and small support headers (`dr_wav`, `pffft`, `nob.h`, `stb_leakcheck.h`, `custom_main.h`).
- `nob.c`: Build script that supports `gcc`, `clang`, `cl`, and `clang-cl`, plus debug and profiling builds.
- `samples/`: Contains sample input audio and impulse response files.
- `scripts/`: Reference scripts and small experiments (`main.py`, `conv1D.py`, `conv2D.py`, `fft.c`).
- `profiling/`: Contains profiling results and a guide on how to reproduce them.

## Building and Running

### Build Instructions

The project uses a "nob" style build system.

1.  **Bootstrapping:**
    ```powershell
    # On Windows (MSVC)
    cl.exe nob.c
    # On Windows (GCC/MinGW)
    gcc nob.c -o nob.exe
    ```
2.  **Building the Project:**
    Run the `nob` executable:
    ```powershell
    .\nob.exe
    ```

Useful build flags:

- `-gcc` (default), `-clang`, `-msvc`, `-clang-cl`
- `-debug` for a debug build
- `-profiling` for profiling-oriented `clang-cl` flags
- `-native` to enable `-march=native` on `gcc`/`clang`
- `-run` to run `build\main.exe` after a successful build

### Running the Application

The executable accepts an optional mode flag (`-m naive`, `-m parallel`, `-m simd`, or `-m fft`).
If omitted, the default mode is `naive`.

Input constraints:

- The input and impulse WAV files must have the same sample rate.
- The impulse must be mono or have the same number of channels as the input.

```powershell
.\build\main.exe <input.wav> <impulse.wav> <output.wav> [-m <naive|parallel|simd|fft>]
```

Example:

```powershell
.\build\main.exe samples\IN_Snare_Classic.wav samples\IR_DocciaAlbergo_44100.wav samples\OUT_Classic_Doccia.wav -m simd
```

## Implementation Details

The convolution is performed either in the time domain or the frequency domain.

**Time Domain:**

$$ y[n] = \sum\_{k=0}^{M-1} x[n-k] \cdot h[k] $$

The `parallel` implementation uses `#pragma omp parallel for` to distribute the outer loop across available CPU cores, while the `naive` version includes a progress callback for real-time feedback.

**Frequency Domain:**
The `fft` mode uses [PFFFT](https://bitbucket.org/jpommier/pffft/) and an overlap-save style block convolution. Signals are transformed into the frequency domain, where convolution becomes a point-wise multiplication:

$$ Y[f] = X[f] \cdot H[f] $$

This is followed by an Inverse FFT to obtain the output signal.
