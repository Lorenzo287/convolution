/*
https://github.com/mackron/dr_libs
https://de-fellows.github.io/RexCoding/python/convolution/2023/06/22/conv_blog.html
https://vincmazet.github.io/bip/filtering/convolution.html
https://www.dspguide.com/ch18/2.htm
*/

#include <stdio.h>
#include <stdlib.h>
#if defined(DEBUG)
    #include "custom_main.h"
#endif

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

typedef void (*ProgressCallback)(size_t current, size_t total);
void update_progress_bar(size_t current, size_t total);

enum { MONO = 0, LEFT = 0, RIGHT = 1 };

void convolve_naive(const float *pInput, size_t inputSize, const float *pKernel,
                    size_t kernelSize, float *pOutput, unsigned int inputChannels,
                    unsigned int kernelChannels, ProgressCallback onProgress) {
    // MACROS for math-like readability
#define X(nk, c) (pInput[(nk) * inputChannels + (c)])
#define H(k, c) \
    ((kernelChannels == 1) ? (pKernel[(k)]) : (pKernel[(k) * kernelChannels + (c)]))
#define Y(n, c) (pOutput[(n) * inputChannels + (c)])

    size_t outputSize = inputSize + kernelSize - 1;
    size_t totalIterations = (size_t)inputChannels * outputSize;
    size_t currentIteration = 0;

    for (size_t n = 0; n < outputSize; n++) {
        // Pre-calculate inner loop bounds
        size_t k_start = (n >= inputSize) ? (n - inputSize + 1) : 0;
        size_t k_end = (n < kernelSize) ? n : (kernelSize - 1);

        if (inputChannels == 2) {
            float sumL = 0.0f;
            float sumR = 0.0f;
            for (size_t k = k_start; k <= k_end; k++) {
                sumL += X(n - k, LEFT) * H(k, LEFT);
                sumR += X(n - k, RIGHT) * H(k, RIGHT);
            }
            Y(n, LEFT) = sumL;
            Y(n, RIGHT) = sumR;
            currentIteration += 2;
        } else {  // mono
            float sum = 0.0f;
            for (size_t k = k_start; k <= k_end; k++) {
                sum += X(n - k, MONO) * H(k, MONO);
            }
            Y(n, MONO) = sum;
            currentIteration++;
        }

        // Only update progress every 1024 samples
        if (onProgress && ((currentIteration % 1024) < inputChannels ||
                           currentIteration >= totalIterations)) {
            onProgress(currentIteration, totalIterations);
        }
    }
#undef X
#undef H
#undef Y
}

int main(int argc, char **argv) {
    int ret = 0;
    if (argc < 4) {
        printf("Usage: %s <input.wav> <impulse.wav> <output.wav>\n", argv[0]);
        ret = -1;
        goto defer;
    }

    const char *inputPath = argv[1];
    const char *impulsePath = argv[2];
    const char *outputPath = argv[3];

    // If the wav is stereo (inputChannels=2) samples are interleaved [ L R L R ... ]
    unsigned int inputChannels, inputSampleRate;
    drwav_uint64 inputFrameCount;
    float *pInputSamples = drwav_open_file_and_read_pcm_frames_f32(
        inputPath, &inputChannels, &inputSampleRate, &inputFrameCount, NULL);

    if (pInputSamples == NULL) {
        fprintf(stderr, "Error: Could not open input file '%s'\n", inputPath);
        ret = -1;
        goto defer;
    }

    unsigned int impulseChannels, impulseSampleRate;
    drwav_uint64 impulseFrameCount;
    float *pImpulseSamples = drwav_open_file_and_read_pcm_frames_f32(
        impulsePath, &impulseChannels, &impulseSampleRate, &impulseFrameCount, NULL);

    if (pImpulseSamples == NULL) {
        fprintf(stderr, "Error: Could not open impulse file '%s'\n", impulsePath);
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

    printf(
        "Processing: %s (%zu samples, %u channels) * %s (%zu samples, %u "
        "channels)...\n",
        inputPath, N, inputChannels, impulsePath, M, impulseChannels);

    float *pOutputSamples = malloc(outSize * outChannels * sizeof(float));
    if (pOutputSamples == NULL) {
        fprintf(stderr, "Error: Out of memory\n");
        ret = -1;
        goto defer_impulse;
    }

    ProgressCallback drawBar = update_progress_bar;
    convolve_naive(pInputSamples, N, pImpulseSamples, M, pOutputSamples,
                   inputChannels, impulseChannels, drawBar);

    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = outChannels;
    format.sampleRate = inputSampleRate;
    format.bitsPerSample = 32;

    drwav wavWrite;
    if (drwav_init_file_write(&wavWrite, outputPath, &format, NULL)) {
        drwav_write_pcm_frames(&wavWrite, outSize, pOutputSamples);
        drwav_uninit(&wavWrite);
        printf("Success! Saved to %s\n", outputPath);
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
    int percent = (int)((float)current / total * 100);

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
