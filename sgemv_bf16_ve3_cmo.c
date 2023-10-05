void sgemv_bf16_ve3_cmo(float *y, float *x, __fp16 *w, int n, int d, int nd) {
    for (int i = 0; i < nd; i++)
        y[i] = 0.0f;
    for (int j = 0; j < n; j++) {
        float tmp = x[j];
        #pragma _NEC ivdep
        for (int i = 0; i < nd; i++) {
            y[i] += w[j * d + i] * tmp;
        }
    }
}
