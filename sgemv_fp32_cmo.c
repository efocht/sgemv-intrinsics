#include <velintrin.h>

#define VLEN (256)

void sgemv_fp32_cmo(float *y, float *x, float *w, int n, int d, int nd) {
    float zero[2] = {0.0f, 0.0f};
    __vr wv1, wv2, wv3, wv4;
    __vr yt1, yt2, yt3, yt4;
    __vr yt;
    int d2 = 2*d;
    int d3 = 3*d;
    int d4 = 4*d;
    for (int i = 0; i < nd; i += VLEN) {
        const int vl = nd - i < VLEN ? nd - i : VLEN;
        yt1 = _vel_vldunc_vssl(0, &zero[0], vl);
        yt2 = _vel_vldunc_vssl(0, &zero[0], vl);
        yt3 = _vel_vldunc_vssl(0, &zero[0], vl);
        yt4 = _vel_vldunc_vssl(0, &zero[0], vl);
    
        float *wp = w + i;
        for (int j = 0; j < n; j+=4) {
            float xs1 = x[j];
            float xs2 = x[j+1];
            float xs3 = x[j+2];
            float xs4 = x[j+3];

            wv1 = _vel_vldunc_vssl(4, wp, vl);
            wv2 = _vel_vldunc_vssl(4, wp + d, vl);
            wv3 = _vel_vldunc_vssl(4, wp + d2, vl);
            wv4 = _vel_vldunc_vssl(4, wp + d3, vl);
            yt1 = _vel_vfmads_vvsvl(yt1, xs1, wv1, vl);
            yt2 = _vel_vfmads_vvsvl(yt2, xs2, wv2, vl);
            yt3 = _vel_vfmads_vvsvl(yt3, xs3, wv3, vl);
            yt4 = _vel_vfmads_vvsvl(yt4, xs4, wv4, vl);
            wp += d4;
        }
        yt2 = _vel_vfadds_vvvl(yt1, yt2, vl);
        yt3 = _vel_vfadds_vvvl(yt3, yt2, vl);
        yt4 = _vel_vfadds_vvvl(yt4, yt3, vl);
        _vel_vstunc_vssl(yt4, 4, (void *)(y+i), vl);
    }
}
