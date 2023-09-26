#include <velintrin.h>

#define VLEN (256)

void sgemv_fp32_simple(float *y, float *x, float *w, int n, int d) {
    int i, vl;
    float zero[2] = {0.0f, 0.0f};
    for (i = 0; i < d; i++) {
        __vr xv;
        __vr wv;
        __vr tv;

        tv = _vel_vldu_vssl(0, &zero[0], VLEN);  // quick way to load zeros

        for (int j = 0; j < n; j += VLEN) {
            const int vl = n - j < VLEN ? n - j : VLEN;

            xv = _vel_vldu_vssl(4, x + j, vl);
            wv = _vel_vldu_vssl(4, w + i * n + j, vl);
            tv = _vel_vfmads_vvvvl(tv, wv, xv, vl);
        }
        tv = _vel_vfsums_vvl(tv, VLEN);
        
        y[i] = _vel_lvss_svs(tv, 0);

  }
}
