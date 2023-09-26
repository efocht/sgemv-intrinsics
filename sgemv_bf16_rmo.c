#include <velintrin.h>

#define VLEN (256)

typedef unsigned short bf16;
typedef union {
    float f[2];
    unsigned long u;
} packed_fp32;

#define load_bf16_to_packed_fp32(wv,wp,vlen) \
    do { \
        wv = _vel_vldunc_vssl(4, (void *)(wp), vlen); \
        __vr wr = _vel_vsrl_vvsl(wv, 16, vlen); \
        wr = _vel_vand_vvvl(wr, bf16mskl, vlen); \
        wv = _vel_vor_vvvl(wv, wr, vlen); \
    } while(0)

void sgemv_bf16_rmo(float *y, float *x, bf16 *w, int n, int d, int nd) {
    float zero[2] = {0.0f, 0.0f};
    __vr wv1, wv2, wv3, wv4;

    __vr bf16mskl = _vel_vbrdl_vsl(0x00000000ffff0000, VLEN);
    __vr low32msk = _vel_vbrdl_vsl(0x00000000ffffffff, VLEN);
    packed_fp32 pf1;

    for (int i = 0; i < nd; i += 2*VLEN) {
        const int vl = nd - i < 2*VLEN ? (nd - i)>>1 : VLEN;
        __vr yt1 = _vel_vld_vssl(0, &zero[0], vl);
        bf16 *wp = w + i;
        for (int j = 0; j < n; j++) {
            pf1.f[0] = pf1.f[1] = x[j];

            load_bf16_to_packed_fp32(wv1, wp, vl);
            yt1 = _vel_pvfmad_vvsvl(yt1, pf1.u, wv1, vl);
            wp += d;
        }
        _vel_vstunc_vssl(yt1, 8, (void *)(y+i+1), vl);
        _vel_vstlnc_vssl(yt1, 8, (void *)(y+i), vl);
    }
}
