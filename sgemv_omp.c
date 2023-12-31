void sgemv_omp(float* xout, float* x, bf16* w, int n, int d) {
    #pragma omp parallel
    {
        int nthr = omp_get_max_threads();
        int ithr = omp_get_thread_num();
        int block = (d + nthr - 1) / nthr;
        int imin = ithr * block;
        int imax = (ithr + 1) * block > d ? d : (ithr + 1) * block;
        sgemv_packed_bf16_unr(&xout[imin], x, &w[imin * n], n, imax - imin);
    }
}
