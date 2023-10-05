#include <stdio.h>
#include <velintrin.h>

#define VLEN (256)
typedef unsigned short bf16;

#define load_bf16_to_packed_fp32(wv,wp,vlen) \
    do { \
        wv = _vel_vldunc_vssl(4, (void *)(wp), vlen); \
        __vr wr = _vel_vsrl_vvsl(wv, 16, vlen); \
        wr = _vel_vand_vvvl(wr, bf16mskl, vlen); \
        wv = _vel_vor_vvvl(wv, wr, vlen); \
    } while(0)

#define sumup_packed_fp32_store(tv,yt,VLEN) \
    do { \
        __vr rv = _vel_vand_vvvl(tv, low32msk, VLEN); \
        __vr sv = _vel_vsll_vvsl(rv, 32, VLEN); \
        tv = _vel_vfadds_vvvl(tv, sv, VLEN); \
        tv = _vel_vfsums_vvl(tv, VLEN); \
        yt = _vel_lvss_svs(tv, 0); \
    } while(0)


void sgemv_packed_bf16_unr(float *y, float *x, bf16 *w, int n, int d) {
    int i;
    float zero[2] = {0.0f, 0.0f};
    __vr bf16mskl = _vel_vbrdl_vsl(0x00000000ffff0000, VLEN);
    __vr low32msk = _vel_vbrdl_vsl(0x00000000ffffffff, VLEN);
    for (i = 0; i < d; i++) {
        __vr xv, xlv;
        __vr wv1;
        __vr tv1 = _vel_vld_vssl(0, &zero[0], VLEN);
        bf16 *wp1 = w + i * n;
        for (int j = 0; j < n; j += 2*VLEN) {
            const int vl = n - j < 2*VLEN ? (n - j)>>1 : VLEN;

            if ((unsigned long)(x+j) & 0x7) {
                xv = _vel_vldu_vssl(8, (void *)(x + j + 1), vl);
                xlv = _vel_vldlzx_vssl(8, (void *)(x + j), vl);
                xv = _vel_pvor_vvvl(xv, xlv, vl);
            } else {
                xv = _vel_vld_vssl(8, (void *)(x + j), vl);
            }
            load_bf16_to_packed_fp32(wv1,wp1+j,vl);
            tv1 = _vel_pvfmad_vvvvl(tv1, xv, wv1, vl);
        }
        sumup_packed_fp32_store(tv1,y[i],VLEN);        
    }
}
