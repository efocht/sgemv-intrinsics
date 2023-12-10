#define CHUNK 4096

// Parallelize along n dimension
void sgemv_cmo_omp2(float* xout, float* x, bfloat16* w, int n, int d) {
    int nthr = omp_get_max_threads();
    float tmp[16][CHUNK];
    for (int i = 0; i < d; i += CHUNK) {
        int dmax = i + CHUNK > d ? d : i + CHUNK;
        #pragma omp parallel
        {
            int ithr = omp_get_thread_num();
            int block = (n + nthr - 1) / nthr;
            int nmin = ithr * block;
            int nmax = (ithr + 1) * block > n ? n : (ithr + 1) * block;
            sgemv_bf16_cmo_n(&tmp, x, &w[imin], n, d, nmin, nmax, i, dmax);
        }
        for (int ithr = 0; ithr < nthr; ithr++) {
            for (int ii = i; ii < dmax; ii++)
                xout[ii] += tmp[ithr][ii - i];
        }
    }
}
