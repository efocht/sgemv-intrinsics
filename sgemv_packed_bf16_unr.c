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

    if (d >= 16) {
        for (i = 0; i < d; i+=16) {
            __vr xv, xlv;
            __vr wv1, wv2, wv3, wv4, wv5, wv6, wv7, wv8;
            __vr wv9, wv10, wv11, wv12, wv13, wv14, wv15, wv16;
        
            __vr tv1 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv2 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv3 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv4 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv5 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv6 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv7 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv8 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv9 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv10 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv11 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv12 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv13 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv14 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv15 = _vel_vld_vssl(0, &zero[0], VLEN);
            __vr tv16 = _vel_vld_vssl(0, &zero[0], VLEN);

            bf16 *wp1 = w + i * n;
            bf16 *wp2 = w + (i + 1) * n;
            bf16 *wp3 = w + (i + 2) * n;
            bf16 *wp4 = w + (i + 3) * n;
            bf16 *wp5 = w + (i + 4) * n;
            bf16 *wp6 = w + (i + 5) * n;
            bf16 *wp7 = w + (i + 6) * n;
            bf16 *wp8 = w + (i + 7) * n;
            bf16 *wp9 = w + (i + 8) * n;
            bf16 *wp10 = w + (i + 9) * n;
            bf16 *wp11 = w + (i + 10) * n;
            bf16 *wp12 = w + (i + 11) * n;
            bf16 *wp13 = w + (i + 12) * n;
            bf16 *wp14 = w + (i + 13) * n;
            bf16 *wp15 = w + (i + 14) * n;
            bf16 *wp16 = w + (i + 15) * n;
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
                load_bf16_to_packed_fp32(wv2,wp2+j,vl);
                load_bf16_to_packed_fp32(wv3,wp3+j,vl);
                load_bf16_to_packed_fp32(wv4,wp4+j,vl);
                load_bf16_to_packed_fp32(wv5,wp5+j,vl);
                load_bf16_to_packed_fp32(wv6,wp6+j,vl);
                load_bf16_to_packed_fp32(wv7,wp7+j,vl);
                load_bf16_to_packed_fp32(wv8,wp8+j,vl);
                load_bf16_to_packed_fp32(wv9,wp9+j,vl);
                load_bf16_to_packed_fp32(wv10,wp10+j,vl);
                load_bf16_to_packed_fp32(wv11,wp11+j,vl);
                load_bf16_to_packed_fp32(wv12,wp12+j,vl);
                load_bf16_to_packed_fp32(wv13,wp13+j,vl);
                load_bf16_to_packed_fp32(wv14,wp14+j,vl);
                load_bf16_to_packed_fp32(wv15,wp15+j,vl);
                load_bf16_to_packed_fp32(wv16,wp16+j,vl);

                tv1 = _vel_pvfmad_vvvvl(tv1, xv, wv1, vl);
                tv2 = _vel_pvfmad_vvvvl(tv2, xv, wv2, vl);
                tv3 = _vel_pvfmad_vvvvl(tv3, xv, wv3, vl);
                tv4 = _vel_pvfmad_vvvvl(tv4, xv, wv4, vl);
                tv5 = _vel_pvfmad_vvvvl(tv5, xv, wv5, vl);
                tv6 = _vel_pvfmad_vvvvl(tv6, xv, wv6, vl);
                tv7 = _vel_pvfmad_vvvvl(tv7, xv, wv7, vl);
                tv8 = _vel_pvfmad_vvvvl(tv8, xv, wv8, vl);
                tv9 = _vel_pvfmad_vvvvl(tv9, xv, wv9, vl);
                tv10 = _vel_pvfmad_vvvvl(tv10, xv, wv10, vl);
                tv11 = _vel_pvfmad_vvvvl(tv11, xv, wv11, vl);
                tv12 = _vel_pvfmad_vvvvl(tv12, xv, wv12, vl);
                tv13 = _vel_pvfmad_vvvvl(tv13, xv, wv13, vl);
                tv14 = _vel_pvfmad_vvvvl(tv14, xv, wv14, vl);
                tv15 = _vel_pvfmad_vvvvl(tv15, xv, wv15, vl);
                tv16 = _vel_pvfmad_vvvvl(tv16, xv, wv16, vl);
            }
            sumup_packed_fp32_store(tv1,y[i],VLEN);        
            sumup_packed_fp32_store(tv2,y[i + 1],VLEN);        
            sumup_packed_fp32_store(tv3,y[i + 2],VLEN);        
            sumup_packed_fp32_store(tv4,y[i + 3],VLEN);        
            sumup_packed_fp32_store(tv5,y[i + 4],VLEN);        
            sumup_packed_fp32_store(tv6,y[i + 5],VLEN);        
            sumup_packed_fp32_store(tv7,y[i + 6],VLEN);        
            sumup_packed_fp32_store(tv8,y[i + 7],VLEN);        
            sumup_packed_fp32_store(tv9,y[i + 8],VLEN);        
            sumup_packed_fp32_store(tv10,y[i + 9],VLEN);        
            sumup_packed_fp32_store(tv11,y[i + 10],VLEN);        
            sumup_packed_fp32_store(tv12,y[i + 11],VLEN);        
            sumup_packed_fp32_store(tv13,y[i + 12],VLEN);        
            sumup_packed_fp32_store(tv14,y[i + 13],VLEN);        
            sumup_packed_fp32_store(tv15,y[i + 14],VLEN);        
            sumup_packed_fp32_store(tv16,y[i + 15],VLEN);        
        }
    }
    for (i = (d/16) * 16; i < d; i++) {
        __vr xv, xlv;
        __vr wv1;
        __vr tv1 = _vel_vld_vssl(0, &zero[0], VLEN);
        bf16 *wp1 = w + i * n;
        for (int j = 0; j < n; j += 2*VLEN) {
            const int vl = n - j < 2*VLEN ? (n - j)>>1 : VLEN;

            if ((unsigned long)(x+j) & 0x7) {
                //printf("x+j is unaligned: %p\n", (void *)(x+j));
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
