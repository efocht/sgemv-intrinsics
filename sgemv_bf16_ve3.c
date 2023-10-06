void sgemv_bf16_ve3(float* xout, float* x, __fp16* w, int n, int d) {
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i += 8) {
        xout[i    ] = 0.0f;
        xout[i + 1] = 0.0f;
        xout[i + 2] = 0.0f;
        xout[i + 3] = 0.0f;
        xout[i + 4] = 0.0f;
        xout[i + 5] = 0.0f;
        xout[i + 6] = 0.0f;
        xout[i + 7] = 0.0f;
        for (int j = 0; j < n; j++) {
            xout[i    ] += w[(i    ) * n + j] * x[j];
            xout[i + 1] += w[(i + 1) * n + j] * x[j];
            xout[i + 2] += w[(i + 2) * n + j] * x[j];
            xout[i + 3] += w[(i + 3) * n + j] * x[j];
            xout[i + 4] += w[(i + 4) * n + j] * x[j];
            xout[i + 5] += w[(i + 5) * n + j] * x[j];
            xout[i + 6] += w[(i + 6) * n + j] * x[j];
            xout[i + 7] += w[(i + 7) * n + j] * x[j];
        }
    }
    for (i = ((d + 7) / 8) * 8; i < d; i++) {
        xout[i] = 0.0f;
        for (int j = 0; j < n; j++) {
            xout[i] += w[i * n + j] * x[j];
        }
    }
}
