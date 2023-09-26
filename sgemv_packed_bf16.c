#include <stdio.h>
#include <velintrin.h>

#define VLEN (512)
typedef unsigned short bf16;

void sgemv_packed_bf16(float *y, float *x, bf16 *w, int n, int d) {
    int i;
    float zero[2] = {0.0f, 0.0f};
    unsigned long bf16mask1 = 0xffff000000000000;
    __vr bfmsk1 = _vel_vld_vssl(0, &bf16mask1, VLEN>>1);
    unsigned long bf16mask2 = 0x00000000ffff0000;
    __vr bfmsk2 = _vel_vld_vssl(0, &bf16mask2, VLEN>>1);
    unsigned long fp32mask = 0xffffffff00000000;
    unsigned long lower32  = 0x00000000ffffffff;
    __vr umsk = _vel_vld_vssl(0, &fp32mask, VLEN>>1);
    __vr lmsk = _vel_vld_vssl(0, &lower32, VLEN>>1);

    //printf("n=%d m=%d\n", n, d);
    for (i = 0; i < d; i++) {
        __vr xv, xlv;
    
        __vr tv = _vel_vld_vssl(0, &zero[0], VLEN>>1);  // quick way to load zeros
        __vr rv = _vel_vld_vssl(0, &zero[0], VLEN>>1);
        __vr sv = _vel_vld_vssl(0, &zero[0], VLEN>>1);

        bf16 *wp = w + i * n;
        // w + i * n + j
        for (int j = 0; j < n; j += VLEN) {
            const int vl = n - j < VLEN ? n - j : VLEN;

            // if (vl & 1) {
            //     printf("vl=%d\n", vl);
            // }
            if ((unsigned long)(x+j) & 0x7) {
                //printf("x+j is unaligned: %p\n", (void *)(x+j));
                xv = _vel_vldu_vssl(8, (void *)(x + j + 1), vl >> 1);
                xlv = _vel_vldlzx_vssl(8, (void *)(x + j), vl >> 1);
                xv = _vel_pvor_vvvl(xv, xlv, vl >> 1);
            } else {
                xv = _vel_vld_vssl(8, (void *)(x + j), vl >> 1);
            }
            // if ((unsigned long)(w + i * n + j) & 0x3) {
            //     printf("w+i*n+j is unaligned: %p\n", (void *)(w+i*n+j));
            // }
            __vr wv = _vel_vldu_vssl(4, (void *)(wp + j), vl>>1);
            __vr wr = _vel_vsrl_vvsl(wv, 16, vl>>1);
            wr = _vel_vand_vvvl(wr, bfmsk2, vl>>1);
            wv = _vel_vor_vvvl(wv, wr, vl>>1);

            tv = _vel_pvfmad_vvvvl(tv, xv, wv, vl>>1);
        }
        rv = _vel_vand_vvvl(tv, lmsk, VLEN>>1);
        sv = _vel_vsll_vvsl(rv, 32, VLEN>>1);
        //tv = _vel_vand_vvvl(tv, umsk, VLEN>>1);
        tv = _vel_vfadds_vvvl(tv, sv, VLEN>>1);
        tv = _vel_vfsums_vvl(tv, VLEN>>1);
        //tv = _vel_vand_vvvl(tv, umsk, VLEN>>1);
        y[i] = _vel_lvss_svs(tv, 0);

    }
}
