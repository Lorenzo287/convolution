/*
https://github.com/mackron/dr_libs
https://de-fellows.github.io/RexCoding/python/convolution/2023/06/22/conv_blog.html
https://vincmazet.github.io/bip/filtering/convolution.html
https://www.dspguide.com/ch18/2.htm
https://github.com/Lorenzo287/convolution
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(DEBUG)
    #include "custom_main.h"
#endif

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

typedef enum {
    MODE_NAIVE,
    MODE_PARALLEL,
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

#define CH_MONO 0
#define CH_LEFT 0
#define CH_RIGHT 1

// MACROS for math-like readability
#define X(nk, ch) (pInput[(nk) * inputChannels + (ch)])
#define H(k, ch) \
    ((kernelChannels == 1) ? (pKernel[(k)]) : (pKernel[(k) * kernelChannels + (ch)]))
#define Y(n, ch) (pOutput[(n) * inputChannels + (ch)])

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
                sumL += X(n - k, CH_LEFT) * H(k, CH_LEFT);
                sumR += X(n - k, CH_RIGHT) * H(k, CH_RIGHT);
            }
            Y(n, CH_LEFT) = sumL;
            Y(n, CH_RIGHT) = sumR;
        } else {  // mono
            float sum = 0.0f;
            for (size_t k = k_start; k <= k_end; k++) {
                sum += X(n - k, CH_MONO) * H(k, CH_MONO);
            }
            Y(n, CH_MONO) = sum;
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
                sumL += X(n - k, CH_LEFT) * H(k, CH_LEFT);
                sumR += X(n - k, CH_RIGHT) * H(k, CH_RIGHT);
            }
            Y(n, CH_LEFT) = sumL;
            Y(n, CH_RIGHT) = sumR;
        } else {
            float sum = 0.0f;
            for (size_t k = k_start; k <= k_end; k++) {
                sum += X(n - k, CH_MONO) * H(k, CH_MONO);
            }
            Y(n, CH_MONO) = sum;
        }
    }
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
    fprintf(
        stderr,
        "Usage: %s <input.wav> <impulse.wav> <output.wav> [-m <naive|parallel>]\n",
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
            } else {
                fprintf(stderr, "Error: Unknown mode '%s'\n", argv[i]);
                return 0;
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            return 0;
        } else if (argv[i][0] == '-') {
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
