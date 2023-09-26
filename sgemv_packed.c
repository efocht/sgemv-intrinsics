#include <stdio.h>
#include <velintrin.h>

#define VLEN (512)

void sgemv_packed(float *y, float *x, float *w, int n, int d) {
    int i;
    float zero[2] = {0.0f, 0.0f};
    unsigned long fp32mask = 0xffffffff00000000;
    unsigned long lower32  = 0x00000000ffffffff;
    //printf("n=%d m=%d\n", n, d);
    __vr umsk = _vel_vld_vssl(0, &fp32mask, VLEN>>1);
    __vr lmsk = _vel_vld_vssl(0, &lower32, VLEN>>1);
    for (i = 0; i < d; i++) {
        __vr xv, xlv;
        __vr wv, wlv;

        __vr tv = _vel_vld_vssl(0, &zero[0], VLEN>>1);  // quick way to load zeros
        __vr rv = _vel_vld_vssl(0, &zero[0], VLEN>>1);
        __vr sv = _vel_vld_vssl(0, &zero[0], VLEN>>1);

        for (int j = 0; j < n; j += VLEN) {
            const int vl = n - j < VLEN ? n - j : VLEN;

            if (vl & 1) {
                printf("vl=%d\n", vl);
            }
            if ((unsigned long)(x+j) & 0x7) {
                //printf("x+j is unaligned: %p\n", (void *)(x+j));
                xv = _vel_vldu_vssl(8, (void *)(x + j + 1), vl >> 1);
                xlv = _vel_vldlzx_vssl(8, (void *)(x + j), vl >> 1);
                xv = _vel_pvor_vvvl(xv, xlv, vl >> 1);
            } else {
                xv = _vel_vld_vssl(8, (void *)(x + j), vl >> 1);
            }
            if ((unsigned long)(w+i*n+j) & 0x7) {
                //printf("w+i*n+j is unaligned: %p\n", (void *)(w+i*n+j));
                //printf("w=%p i=%d j=%d\n", (void *)w, i, j);
                wv = _vel_vldu_vssl(8, (void *)(w + i * n + j + 1), vl >> 1);
                wlv = _vel_vldlzx_vssl(8, (void *)(w + i * n + j), vl >> 1);
                wv = _vel_pvor_vvvl(wv, wlv, vl >> 1);
            } else {
                wv = _vel_vld_vssl(8, (void *)(w + i * n + j), vl >> 1);
            }

            tv = _vel_pvfmad_vvvvl(tv, xv, wv, vl>>1);
        }
        rv = _vel_vand_vvvl(tv, lmsk, VLEN>>1);
        sv = _vel_vsll_vvsl(rv, 32, VLEN>>1);
        tv = _vel_vand_vvvl(tv, umsk, VLEN>>1);
        tv = _vel_vfadds_vvvl(tv, sv, VLEN>>1);
        tv = _vel_vfsums_vvl(tv, VLEN>>1);
        tv = _vel_vand_vvvl(tv, umsk, VLEN>>1);
        y[i] = _vel_lvss_svs(tv, 0);

    }
}
