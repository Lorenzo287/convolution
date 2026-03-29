# Convolution Project

A C-based implementation of naive time-domain convolution for audio files, specifically targeting the WAV format.

## Project Overview

This project implements a simple convolution engine that applies an Impulse Response (IR) to an input audio signal. It's primarily designed for educational purposes and testing basic audio DSP concepts.

- **Main Technologies:** C, [dr_wav](https://github.com/mackron/dr_libs) for audio I/O, and [nob.h](https://github.com/tsoding/nob.h) for the build system.
- **Key Features:**
    - Support for mono and stereo WAV files (interleaved).
    - Naive and **Parallel (OpenMP)** convolution implementations.
    - High-resolution timing for performance benchmarking.
    - Real-time progress bar with throttled updates for the naive implementation.
    - Automatic build system using `nob.c`.
    - Memory leak detection via `stb_leakcheck.h` (optional).

## Performance Optimizations

Several optimization techniques have been applied to the convolution engine:
- **Parallelization (OpenMP):** Utilizes multi-core processing to distribute the workload, achieving a ~5x speedup on typical hardware.
- **Interleaved Cache Locality:** Loops process all channels in a single pass. Since WAV files are interleaved (L, R, L, R...), this ensures sequential memory access, significantly reducing cache misses.
- **Branchless Inner Loop:** Loop bounds are pre-calculated for every output sample, removing conditional checks from the critical path.
- **Macro-based Sampling:** Uses zero-cost pre-processor macros (e.g., `X(n, c)`) for "math-like" readability of interleaved samples.

## Future Improvements

To further improve performance, the following techniques could be implemented:
- **FFT Convolution:** Transitioning to the frequency domain using the Fast Fourier Transform (Overlap-Add/Save) to reduce complexity from $O(N \cdot M)$ to $O(N \log N)$.
- **SIMD (Vectorization):** Using SSE/AVX intrinsics to perform multiple multiply-accumulate operations in a single clock cycle.

## Project Structure

- `main.c`: Core logic for loading WAV files, convolution engines, and performance timing.
- `nob.c`: Build script that handles compilation with OpenMP support.
- `include/`:
    - `dr_wav.h`: WAV file handling library.
    - `stb_leakcheck.h`: Utility for detecting memory leaks.
    - `custom_main.h`: Debug wrapper for `main()`.
- `test/`: Contains sample input audio and impulse response files.

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

### Running the Application

The executable accepts an optional mode flag (`-m naive` or `-m parallel`).

```powershell
.\build\main.exe <input.wav> <impulse.wav> <output.wav> [-m <naive|parallel>]
```

Example:
```powershell
.\build\main.exe test\IN_Snare_Classic.wav test\IR_DocciaAlbergo_44100.wav test\OUT_Classic_Doccia.wav -m parallel
```

## Development Conventions

- **Audio I/O:** Always use `dr_wav` for reading and writing audio.
- **OpenMP:** Maintain thread-safety in new convolution functions.
- **UI/DSP Separation:** Keep DSP functions independent of UI logic.
- **Interleaving:** Handle interleaved channels by processing all channels simultaneously to maximize cache efficiency.

## Implementation Details

The convolution is performed in the time domain:

$$ y[n] = \sum_{k=0}^{M-1} x[n-k] \cdot h[k] $$

The `parallel` implementation uses `#pragma omp parallel for` to distribute the outer loop across available CPU cores, while the `naive` version includes a progress callback for real-time feedback.
