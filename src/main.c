/*
https://github.com/mackron/dr_libs
https://de-fellows.github.io/RexCoding/python/convolution/2023/06/22/conv_blog.html
https://vincmazet.github.io/bip/filtering/convolution.html
https://www.dspguide.com/ch18/2.htm
https://github.com/Lorenzo287/convolution
*/

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(DEBUG)
    #include "custom_main.h"
#endif

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include "pffft.h"

typedef enum {
    MODE_NAIVE,
    MODE_PARALLEL,
    MODE_SIMD,
    MODE_FFT,
} ConvMode;

typedef struct {
    const char *inputPath;
    const char *impulsePath;
    const char *outputPath;
    ConvMode mode;
} Config;

void print_usage(const char *progName);
int parse_args(int argc, char **argv, Config *config);
typedef void (*ProgressCallback)(size_t current, size_t total);
void update_progress_bar(size_t current, size_t total);

// MACROS for math-like readability
#define X(nk, ch) (pInput[(nk) * inputChannels + (ch)])
#define H(k, ch) \
    ((kernelChannels == 1) ? (pKernel[(k)]) : (pKernel[(k) * kernelChannels + (ch)]))
#define Y(n, ch) (pOutput[(n) * inputChannels + (ch)])

#define CH_LEFT 0
#define CH_RIGHT 1
#define CH_MONO CH_LEFT

#define XM(nk) X(nk, CH_MONO)
#define XL(nk) X(nk, CH_LEFT)
#define XR(nk) X(nk, CH_RIGHT)
#define HM(n) H(n, CH_MONO)
#define HL(n) H(n, CH_LEFT)
#define HR(n) H(n, CH_RIGHT)
#define YM(n) Y(n, CH_MONO)
#define YL(n) Y(n, CH_LEFT)
#define YR(n) Y(n, CH_RIGHT)

void convolve_naive(const float *pInput, size_t inputSize, const float *pKernel,
                    size_t kernelSize, float *pOutput, unsigned int inputChannels,
                    unsigned int kernelChannels, ProgressCallback onProgress) {
    size_t outputSize = inputSize + kernelSize - 1;

    for (size_t n = 0; n < outputSize; n++) {
        // Pre-calculate inner loop bounds
        size_t k_start = (n >= inputSize) ? (n - inputSize + 1) : 0;
        size_t k_end = (n < kernelSize) ? n : (kernelSize - 1);

        if (inputChannels == 2) {
            float sumL = 0.0f;
            float sumR = 0.0f;
            for (size_t k = k_start; k <= k_end; k++) {
                sumL += XL(n - k) * HL(k);
                sumR += XR(n - k) * HR(k);
            }
            YL(n) = sumL;
            YR(n) = sumR;
        } else {  // mono
            float sum = 0.0f;
            for (size_t k = k_start; k <= k_end; k++) { sum += XM(n - k) * HM(k); }
            YM(n) = sum;
        }
        // Only update progress every 2048 samples
        if (onProgress && (n % 2048 == 0)) onProgress(n, outputSize);
    }
    if (onProgress) onProgress(outputSize, outputSize);
}

void convolve_parallel(const float *pInput, size_t inputSize, const float *pKernel,
                       size_t kernelSize, float *pOutput, unsigned int inputChannels,
                       unsigned int kernelChannels) {
    size_t outputSize = inputSize + kernelSize - 1;

#pragma omp parallel for schedule(static)
    for (size_t n = 0; n < outputSize; n++) {
        size_t k_start = (n >= inputSize) ? (n - inputSize + 1) : 0;
        size_t k_end = (n < kernelSize) ? n : (kernelSize - 1);

        if (inputChannels == 2) {
            float sumL = 0.0f;
            float sumR = 0.0f;
            for (size_t k = k_start; k <= k_end; k++) {
                sumL += XL(n - k) * HL(k);
                sumR += XR(n - k) * HR(k);
            }
            YL(n) = sumL;
            YR(n) = sumR;
        } else {
            float sum = 0.0f;
            for (size_t k = k_start; k <= k_end; k++) { sum += XM(n - k) * HM(k); }
            YM(n) = sum;
        }
    }
}

// collapse vector [a,b,c,d,e,f,g,h] down to a+b+c+d+e+f+g+h
static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);    // lower 4 floats
    __m128 hi = _mm256_extractf128_ps(v, 1);  // upper 4 floats
    lo = _mm_add_ps(lo, hi);                  // add pairwise → 4 floats
    lo = _mm_hadd_ps(lo, lo);                 // horizontal add → 2 floats
    lo = _mm_hadd_ps(lo, lo);                 // horizontal add → 1 float
    return _mm_cvtss_f32(lo);                 // extract scalar
}

void convolve_simd(const float *pInput, size_t inputSize, const float *pKernel,
                   size_t kernelSize, float *pOutput, unsigned int inputChannels,
                   unsigned int kernelChannels) {
    size_t outputSize = inputSize + kernelSize - 1;
    size_t paddedInputSize = inputSize + kernelSize + 8;

    // Allocate buffers to de-interleave input and kernel,
    // and add padding (zeros -> calloc) at the beginning and end,
    // so that when loading 8 floats at a time it does not go out of bounds
    float *pInputL = calloc(paddedInputSize, sizeof(float));
    if (!pInputL) {
        fprintf(stderr, "Error: Out of memory (pInputL)\n");
        goto defer;
    }
    float *pInputR = NULL;
    if (inputChannels == 2) {
        pInputR = calloc(paddedInputSize, sizeof(float));
        if (!pInputR) {
            fprintf(stderr, "Error: Out of memory (pInputR)\n");
            goto defer_inputL;
        }
    }
    float *pKernelL = calloc(kernelSize + 8, sizeof(float));
    if (!pKernelL) {
        fprintf(stderr, "Error: Out of memory (pKernelL)\n");
        goto defer_inputR;
    }
    float *pKernelR = NULL;
    if (inputChannels == 2) {
        pKernelR = calloc(kernelSize + 8, sizeof(float));
        if (!pKernelR) {
            fprintf(stderr, "Error: Out of memory (pKernelR)\n");
            goto defer_kernelL;
        }
    }

    // Copy de-interleaved input, with padding at start and end
    for (size_t i = 0; i < inputSize; i++) {
        pInputL[i + kernelSize - 1] = pInput[i * inputChannels + 0];
        if (inputChannels == 2)
            pInputR[i + kernelSize - 1] = pInput[i * inputChannels + 1];
    }
    // Copy de-interleaved REVERSED kernel, so that the conv becomes a dot product
    for (size_t i = 0; i < kernelSize; i++) {
        pKernelL[i] = pKernel[(kernelSize - 1 - i) * kernelChannels + 0];
        if (kernelChannels == 2)
            pKernelR[i] = pKernel[(kernelSize - 1 - i) * kernelChannels + 1];
        else if (inputChannels == 2)
            pKernelR[i] = pKernelL[i];
    }

    if (inputChannels == 2) {
#pragma omp parallel for schedule(static)
        for (size_t n = 0; n < outputSize; n++) {
            __m256 sumL_vec = _mm256_setzero_ps();
            __m256 sumR_vec = _mm256_setzero_ps();
            const float *inL = &pInputL[n];
            const float *inR = &pInputR[n];
            size_t k = 0;
            for (; k + 7 < kernelSize; k += 8) {
                sumL_vec = _mm256_fmadd_ps(_mm256_loadu_ps(&inL[k]),
                                           _mm256_loadu_ps(&pKernelL[k]), sumL_vec);
                sumR_vec = _mm256_fmadd_ps(_mm256_loadu_ps(&inR[k]),
                                           _mm256_loadu_ps(&pKernelR[k]), sumR_vec);
            }
            float sumL = hsum256_ps(sumL_vec);
            float sumR = hsum256_ps(sumR_vec);
            for (; k < kernelSize; k++) {
                sumL += inL[k] * pKernelL[k];
                sumR += inR[k] * pKernelR[k];
            }
            pOutput[n * 2 + 0] = sumL;
            pOutput[n * 2 + 1] = sumR;
        }
    } else {
#pragma omp parallel for schedule(static)
        for (size_t n = 0; n < outputSize; n++) {
            __m256 sumL_vec = _mm256_setzero_ps();  // 8 lanes of 0.0f
            const float *inL = &pInputL[n];

            size_t k = 0;
            for (; k + 7 < kernelSize; k += 8) {
                sumL_vec = _mm256_fmadd_ps(  // NOTE: Fused Multiply-Add
                                             // on 265bit (8 * 32bit floats) register
                    _mm256_loadu_ps(&inL[k]),       // load 8 input floats
                    _mm256_loadu_ps(&pKernelL[k]),  // load 8 kernel floats
                    sumL_vec);                      // accumulate
            }
            float sumL = hsum256_ps(sumL_vec);  // collapse 8 lanes → 1 float

            for (; k < kernelSize; k++)        // remainder (remaining samples
                sumL += inL[k] * pKernelL[k];  // when kernelSize % 8 != 0)
            pOutput[n] = sumL;
        }
    }

    free(pKernelR);
defer_kernelL:
    free(pKernelL);
defer_inputR:
    free(pInputR);
defer_inputL:
    free(pInputL);
defer:
    return;
}

static size_t find_next_pffft_size(size_t n) {
    if (n < 32) return 32;
    size_t res = 32;
    while (res < n) res <<= 1;
    return res;
}

void convolve_fft(const float *pInput, size_t inputSize, const float *pKernel,
                  size_t kernelSize, float *pOutput, unsigned int inputChannels,
                  unsigned int kernelChannels) {
    size_t minSize = inputSize + kernelSize - 1;
    size_t fftSize = find_next_pffft_size(minSize);

    PFFFT_Setup *setup = pffft_new_setup((int)fftSize, PFFFT_REAL);
    if (!setup) {
        fprintf(stderr, "Error: Could not create PFFFT setup for size %zu\n", fftSize);
        return;
    }

    float *in_buf = pffft_aligned_malloc(fftSize * sizeof(float));
    float *krn_buf = pffft_aligned_malloc(fftSize * sizeof(float));
    float *out_buf = pffft_aligned_malloc(fftSize * sizeof(float));
    float *in_fft = pffft_aligned_malloc(fftSize * sizeof(float));
    float *krn_fft = pffft_aligned_malloc(fftSize * sizeof(float));
    float *out_fft = pffft_aligned_malloc(fftSize * sizeof(float));
    float *work = pffft_aligned_malloc(fftSize * sizeof(float));

    if (!in_buf || !krn_buf || !out_buf || !in_fft || !krn_fft || !out_fft || !work) {
        fprintf(stderr, "Error: Out of memory for FFT buffers\n");
        goto cleanup;
    }

    for (unsigned int ch = 0; ch < inputChannels; ch++) {
        memset(in_buf, 0, fftSize * sizeof(float));
        memset(krn_buf, 0, fftSize * sizeof(float));
        memset(out_fft, 0, fftSize * sizeof(float));

        for (size_t i = 0; i < inputSize; i++) in_buf[i] = pInput[i * inputChannels + ch];

        unsigned int kCh = (kernelChannels == 1) ? 0 : ch;
        for (size_t i = 0; i < kernelSize; i++)
            krn_buf[i] = pKernel[i * kernelChannels + kCh];

        pffft_transform(setup, in_buf, in_fft, work, PFFFT_FORWARD);
        pffft_transform(setup, krn_buf, krn_fft, work, PFFFT_FORWARD);

        pffft_zconvolve_accumulate(setup, in_fft, krn_fft, out_fft, 1.0f / (float)fftSize);

        pffft_transform(setup, out_fft, out_buf, work, PFFFT_BACKWARD);

        size_t outLen = inputSize + kernelSize - 1;
        for (size_t i = 0; i < outLen; i++) {
            pOutput[i * inputChannels + ch] = out_buf[i];
        }
    }

cleanup:
    if (in_buf) pffft_aligned_free(in_buf);
    if (krn_buf) pffft_aligned_free(krn_buf);
    if (out_buf) pffft_aligned_free(out_buf);
    if (in_fft) pffft_aligned_free(in_fft);
    if (krn_fft) pffft_aligned_free(krn_fft);
    if (out_fft) pffft_aligned_free(out_fft);
    if (work) pffft_aligned_free(work);
    if (setup) pffft_destroy_setup(setup);
}

int main(int argc, char **argv) {
    int ret = 0;
    Config config = {0};

    if (!parse_args(argc, argv, &config)) {
        print_usage(argv[0]);
        ret = -1;
        goto defer;
    }

    // If the wav is stereo (inputChannels=2) samples are interleaved [ L R L R ... ]
    unsigned int inputChannels, inputSampleRate;
    drwav_uint64 inputFrameCount;
    float *pInputSamples = drwav_open_file_and_read_pcm_frames_f32(
        config.inputPath, &inputChannels, &inputSampleRate, &inputFrameCount, NULL);

    if (pInputSamples == NULL) {
        fprintf(stderr, "Error: Could not open input file '%s'\n", config.inputPath);
        ret = -1;
        goto defer;
    }

    unsigned int impulseChannels, impulseSampleRate;
    drwav_uint64 impulseFrameCount;
    float *pImpulseSamples = drwav_open_file_and_read_pcm_frames_f32(
        config.impulsePath, &impulseChannels, &impulseSampleRate, &impulseFrameCount,
        NULL);

    if (pImpulseSamples == NULL) {
        fprintf(stderr, "Error: Could not open impulse file '%s'\n",
                config.impulsePath);
        ret = -1;
        goto defer_input;
    }

    if (impulseChannels != 1 && impulseChannels != inputChannels) {
        fprintf(stderr, "Error: Impulse must be mono or match input channels\n");
        ret = -1;
        goto defer_impulse;
    }

    // It would still work but reverb would be stretched -> pitched up/down
    if (inputSampleRate != impulseSampleRate) {
        fprintf(stderr, "Error: Sample rates do not match! Input: %u, Impulse: %u\n",
                inputSampleRate, impulseSampleRate);
        ret = -1;
        goto defer_impulse;
    }

    size_t N = (size_t)inputFrameCount;
    size_t M = (size_t)impulseFrameCount;
    size_t outSize = N + M - 1;
    unsigned int outChannels = inputChannels;

    float *pOutputSamples = malloc(outSize * outChannels * sizeof(float));
    if (pOutputSamples == NULL) {
        fprintf(stderr, "Error: Out of memory\n");
        ret = -1;
        goto defer_impulse;
    }

    const char *modeStr = "Unknown";
    switch (config.mode) {
    case MODE_NAIVE:
        modeStr = "Naive";
        break;
    case MODE_PARALLEL:
        modeStr = "Parallel (OpenMP)";
        break;
    case MODE_SIMD:
        modeStr = "SIMD (AVX2)";
        break;
    case MODE_FFT:
        modeStr = "FFT (PFFFT)";
        break;
    }
    printf("Mode: %s\n", modeStr);
    printf("Processing: (%zu samples, %u channels) * (%zu samples, %u channels)\n",
           N, inputChannels, M, impulseChannels);

    clock_t start = clock();
    switch (config.mode) {
    case MODE_PARALLEL:
        convolve_parallel(pInputSamples, N, pImpulseSamples, M, pOutputSamples,
                          inputChannels, impulseChannels);
        break;
    case MODE_SIMD:
        convolve_simd(pInputSamples, N, pImpulseSamples, M, pOutputSamples,
                      inputChannels, impulseChannels);
        break;
    case MODE_FFT:
        convolve_fft(pInputSamples, N, pImpulseSamples, M, pOutputSamples,
                     inputChannels, impulseChannels);
        break;
    case MODE_NAIVE:
    default:
        convolve_naive(pInputSamples, N, pImpulseSamples, M, pOutputSamples,
                       inputChannels, impulseChannels, update_progress_bar);
        break;
    }
    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken: %.4f seconds\n", cpu_time_used);

    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = outChannels;
    format.sampleRate = inputSampleRate;
    format.bitsPerSample = 32;

    drwav wavWrite;
    if (drwav_init_file_write(&wavWrite, config.outputPath, &format, NULL)) {
        drwav_write_pcm_frames(&wavWrite, outSize, pOutputSamples);
        drwav_uninit(&wavWrite);
    } else {
        fprintf(stderr, "Error: Could not open output file for writing\n");
    }

    free(pOutputSamples);
defer_impulse:
    drwav_free(pImpulseSamples, NULL);
defer_input:
    drwav_free(pInputSamples, NULL);
defer:
    return ret;
}

void update_progress_bar(size_t current, size_t total) {
    if (total == 0) return;
    static int lastPercent = -1;
    int percent = (int)((float)current / total * 100.0f);

    if (percent != lastPercent || current == total) {
        if (lastPercent == -1) printf("\033[?25l");  // Hide cursor on first call

        lastPercent = percent;
        int barWidth = 40;
        float progress = (float)current / total;
        printf("\r[");
        int pos = (int)(barWidth * progress);
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos)
                printf("=");
            else if (i == pos)
                printf(">");
            else
                printf(" ");
        }
        printf("] %d%%", percent);
        fflush(stdout);

        if (current == total) {
            printf("\033[?25h\n");  // Show cursor and newline on completion
            lastPercent = -1;       // Reset for next run
        }
    }
}

void print_usage(const char *progName) {
    fprintf(stderr,
            "Usage: %s <input.wav> <impulse.wav> <output.wav> [-m "
            "<naive|parallel|simd|fft>]\n",
            progName);
}

int parse_args(int argc, char **argv, Config *config) {
    config->mode = MODE_NAIVE;
    config->inputPath = NULL;
    config->impulsePath = NULL;
    config->outputPath = NULL;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: -m requires an argument\n");
                return 0;
            }
            if (strcmp(argv[i], "naive") == 0) {
                config->mode = MODE_NAIVE;
            } else if (strcmp(argv[i], "parallel") == 0) {
                config->mode = MODE_PARALLEL;
            } else if (strcmp(argv[i], "simd") == 0) {
                config->mode = MODE_SIMD;
            } else if (strcmp(argv[i], "fft") == 0) {
                config->mode = MODE_FFT;
            } else {
                fprintf(stderr, "Error: Unknown mode '%s'\n", argv[i]);
                return 0;
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            return 0;
        } else if (argv[i][0] == '-') {  // followed by char != m
            fprintf(stderr, "Error: Unknown argument '%s'\n", argv[i]);
            return 0;
        } else {
            if (config->inputPath == NULL) {
                config->inputPath = argv[i];
            } else if (config->impulsePath == NULL) {
                config->impulsePath = argv[i];
            } else if (config->outputPath == NULL) {
                config->outputPath = argv[i];
            } else {
                fprintf(stderr, "Error: Too many positional arguments\n");
                return 0;
            }
        }
    }
    if (config->inputPath == NULL || config->impulsePath == NULL ||
        config->outputPath == NULL) {
        return 0;
    }
    return 1;
}
