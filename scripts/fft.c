#include <stdio.h>
#include <math.h>

#define MAX 200
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

/* Complex number structure */
typedef struct {
    double real;
    double imag;
} Complex;

/* Forward declarations for helpers */
Complex complex_mul(Complex a, Complex b);
Complex complex_add(Complex a, Complex b);
Complex complex_sub(Complex a, Complex b);
Complex complex_scale(Complex a, double s);
Complex polar_unit(double angle);
Complex complex_pow(Complex base, int exp);
int log2_int(int N);
int check_power_of_two(int n);
int bit_reverse(int N, int n);

/* ---------- FFT functions ---------- */

/* Reorders the array in bit-reversed order */
void reorder(Complex *f1, int N) {
    Complex f2[MAX];
    for (int i = 0; i < N; i++) f2[i] = f1[bit_reverse(N, i)];
    for (int j = 0; j < N; j++) f1[j] = f2[j];
}

/* Computes the FFT transform in-place */
void transform(Complex *f, int N) {
    reorder(f, N);

    Complex W[MAX / 2];
    W[0] = (Complex){1.0, 0.0};
    W[1] = polar_unit(-2.0 * M_PI / N);
    for (int i = 2; i < N / 2; i++) W[i] = complex_pow(W[1], i);

    int n = 1;
    int a = N / 2;
    for (int j = 0; j < log2_int(N); j++) {
        for (int i = 0; i < N; i++) {
            if (!(i & n)) {
                Complex temp = f[i];
                Complex Temp = complex_mul(W[(i * a) % (n * a)], f[i + n]);
                f[i] = complex_add(temp, Temp);
                f[i + n] = complex_sub(temp, Temp);
            }
        }
        n *= 2;
        a /= 2;
    }
}

/* Applies the FFT and scales by the sampling step d */
void FFT(Complex *f, int N, double d) {
    transform(f, N);
    for (int i = 0; i < N; i++) f[i] = complex_scale(f[i], d);
}

/* ---------- main ---------- */

int main(void) {
    int n;
    do {
        printf("Enter the vector size (must be a power of 2): ");
        scanf("%d", &n);
    } while (!check_power_of_two(n));

    double d;
    printf("Enter the sampling step size: ");
    scanf("%lf", &d);
    printf("Sampling step = %f\n", d);

    Complex vec[MAX];
    printf("Enter the sampling vector:\n");
    for (int i = 0; i < n; i++) {
        double re, im;
        printf("Enter component at index %d (real imag): ", i);
        scanf("%lf %lf", &re, &im);
        vec[i] = (Complex){re, im};
        printf("Index %d = (%f, %f)\n", i, vec[i].real, vec[i].imag);
    }

    FFT(vec, n, d);

    printf("Transformed vector:\n");
    for (int j = 0; j < n; j++) printf("(%f, %f)\n", vec[j].real, vec[j].imag);

    return 0;
}

/* ---------- helpers ---------- */

Complex complex_mul(Complex a, Complex b) {
    return (Complex){a.real * b.real - a.imag * b.imag,
                     a.real * b.imag + a.imag * b.real};
}

Complex complex_add(Complex a, Complex b) {
    return (Complex){a.real + b.real, a.imag + b.imag};
}

Complex complex_sub(Complex a, Complex b) {
    return (Complex){a.real - b.real, a.imag - b.imag};
}

Complex complex_scale(Complex a, double s) {
    return (Complex){a.real * s, a.imag * s};
}

/* polar(1, angle) — unit complex number at given angle in radians */
Complex polar_unit(double angle) {
    return (Complex){cos(angle), sin(angle)};
}

/* Complex power: base^exp (integer exponent) */
Complex complex_pow(Complex base, int exp) {
    Complex result = {1.0, 0.0};
    for (int i = 0; i < exp; i++) result = complex_mul(result, base);
    return result;
}

/* Returns floor(log2(N)) */
int log2_int(int N) {
    int k = N, i = 0;
    while (k) {
        k >>= 1;
        i++;
    }
    return i - 1;
}

/* Returns 1 if n is a power of 2, 0 otherwise */
int check_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

/* Computes the bit-reversal of n with respect to log2(N) bits */
int bit_reverse(int N, int n) {
    int j, p = 0;
    for (j = 1; j <= log2_int(N); j++) {
        if (n & (1 << (log2_int(N) - j))) p |= 1 << (j - 1);
    }
    return p;
}
