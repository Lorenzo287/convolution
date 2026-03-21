# Convolution Project

A C-based implementation of naive time-domain convolution for audio files, specifically targeting the WAV format.

## Project Overview

This project implements a simple convolution engine that applies an Impulse Response (IR) to an input audio signal. It's primarily designed for educational purposes and testing basic audio DSP concepts.

- **Main Technologies:** C, [dr_wav](https://github.com/mackron/dr_libs) for audio I/O, and [nob.h](https://github.com/tsoding/nob.h) for the build system.
- **Key Features:**
    - Support for mono and stereo WAV files (interleaved).
    - Naive time-domain convolution implementation.
    - Real-time progress bar with throttled updates (every 1024 samples) for performance.
    - Decoupled DSP and UI logic using a callback-based architecture.
    - Automatic build system using `nob.c`.
    - Memory leak detection via `stb_leakcheck.h` (optional).

## Performance Optimizations

During the latest development session, several "micro-optimizations" were applied to the naive algorithm:
- **Branchless Inner Loop:** The loop bounds for the kernel are pre-calculated for every output sample, removing a conditional `if` check from the billion-iteration inner loop.
- **Throttled Progress Callback:** The UI update is now triggered only once every 1024 iterations, significantly reducing function call overhead and float-to-int conversions.
- **Macro-based Sampling:** Replaced function calls with pre-processor macros to ensure zero-cost abstraction for interleaved sample access.

## Future Improvements

To further improve performance, the following techniques could be implemented:
- **FFT Convolution:** Transitioning to the frequency domain using the Fast Fourier Transform (Overlap-Add/Save) to reduce complexity from $O(N \cdot M)$ to $O(N \log N)$.
- **Parallelization:** Utilizing **OpenMP** or threads to process audio channels or chunks in parallel.
- **Cache Locality:** Refactoring the loops to process interleaved channels simultaneously (L+R in one pass) rather than iterating through the entire file one channel at a time.
- **SIMD (Vectorization):** Using SSE/AVX intrinsics to perform multiple multiply-accumulate operations in a single clock cycle.

## Project Structure

- `main.c`: Core logic for loading WAV files, performing convolution, and saving the result. Includes a decoupled terminal progress bar.
- `nob.c`: Build script that handles compilation and linking.
- `include/`:
    - `dr_wav.h`: WAV file handling library.
    - `stb_leakcheck.h`: Utility for detecting memory leaks.
    - `custom_main.h`: Debug wrapper for `main()` to enable leak checking.
- `test/`: Contains sample input audio and impulse response files.
- `conv1.py`, `conv2.py`: Python scripts for reference/verification (1D and 2D convolution).

## Building and Running

### Build Instructions

The project uses a "nob" style build system.

1.  **Bootstrapping (if `nob.exe` is missing):**
    ```powershell
    # On Windows (MSVC)
    cl.exe nob.c
    # On Linux/macOS (GCC/Clang)
    gcc -o nob nob.c
    ```
2.  **Building the Project:**
    Run the `nob` executable:
    ```powershell
    .\nob.exe
    ```
    This will compile `main.c` into the `build/` directory.

### Running the Application

The compiled executable expects three arguments: input file, impulse response, and output path.

```powershell
.\build\main.exe test\IN_Snare_Classic.wav test\IR_DocciaAlbergo_44100.wav test\OUT_Classic_Doccia.wav
```

## Development Conventions

- **Audio I/O:** Always use `dr_wav` for reading and writing audio.
- **Sample Rates:** The input and impulse response must have matching sample rates.
- **Memory Management:** Use `drwav_free` for `dr_wav` allocations and standard `free` for others. When `DEBUG` is defined, `stb_leakcheck` is used to track allocations.
- **UI/DSP Separation:** Keep DSP functions (like `convolve_naive`) independent of UI logic. Use the `ProgressCallback` pattern for reporting status.
- **Interleaving:** The convolution logic handles interleaved channels by iterating through each channel separately.

## Implementation Details

The convolution is performed in the time domain:
$$ y[n] = \sum_{k=0}^{M-1} x[n-k] \cdot h[k] $$
Where:
- $x$ is the input signal.
- $h$ is the impulse response of length $M$.
- $y$ is the output signal.

The implementation in `main.c` uses a direct loop that iterates over each output sample and calculates the weighted sum. To ensure high performance, the kernel indices ($k$) are pre-calculated for each $n$ to stay within the bounds of the input signal, eliminating the need for branching inside the most critical part of the code.
