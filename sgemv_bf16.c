#include <velintrin.h>

#define VLEN (256)

void sgemv_bf16(float *y, float *x, unsigned short *w, int n, int d) {
    int i;
    float zero[2] = {0.0f, 0.0f};
    unsigned long bf16mask = 0xffff000000000000;

    __vr bfmsk = _vel_vld_vssl(0, &bf16mask, VLEN);

    // broadcast 0 1 2 3 4 ... VLEN-1
    __vr seq = _vel_vseq_vl(VLEN);
    // shift 1 right
    __vr indx = _vel_vsrl_vvsl(seq, 1, VLEN);
    // odd indices contain a 1
    __vr odd = _vel_vand_vsvl(1, seq, VLEN);
    // vector mask for left shifts
    __vm256 oddmsk = _vel_vfmklne_mvl(odd, VLEN);
    __vm256 evenmsk = _vel_negm_mm(oddmsk);

    #pragma clang loop unroll_count(4)
    for (i = 0; i < d; i++) {
        __vr xv;
        __vr wv;
        __vr tv;

        tv = _vel_vldu_vssl(0, &zero[0], VLEN);  // quick way to load zeros

        for (int j = 0; j < n; j += VLEN) {
            const int vl = n - j < VLEN ? n - j : VLEN;

            xv = _vel_vldu_vssl(4, x + j, vl);

            //wv = _vel_vldu_vssl(4, w + i * n + j, vl);
            __vr aidx = _vel_vsfa_vvssl(indx, 2, (unsigned long)(w + i * n + j), vl);
            wv = _vel_vgtu_vvssl(aidx, 0, 0, vl);

            // shift left the odd entries
            wv = _vel_vsll_vvsmvl(wv, 16, evenmsk, wv, vl);

            wv = _vel_vand_vvvl(wv, bfmsk, vl);

            tv = _vel_vfmads_vvvvl(tv, wv, xv, vl);
        }
        tv = _vel_vfsums_vvl(tv, VLEN);
        
        y[i] = _vel_lvss_svs(tv, 0);

  }
}
