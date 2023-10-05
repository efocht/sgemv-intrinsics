## BFLOAT16 SGEMV with LLVM intrinsics

This repository contains `bfloat16` implementations of a matrix-vector multiply
for the SX-Aurora Vector Engine (VE) generation 1 and 2. These VEs do not support
the `bfloat16` floating point format!

The two relevant function prototypes are:
```
void sgemv_packed_bf16_unr(float *y, float *x, bf16 *w, int n, int d);
```
This function implements `y = w * x`, with `w` being a dense matrix of dimension `d x n`,
`x` a vector of dimension `n` and `y` the result vector of dimension `d`. The matrix is
stored in *row memory order*.

```
void sgemv_bf16_cmo(float *y, float *x, bf16 *w, int n, int d, int nd);
```
This function implements `y = w * x`, with `w` being a dense matrix of dimension `d x n`,
`x` a vector of dimension `n` and `y` the result vector of dimension `d`. The matrix is
stored in *column memory order* (!) and the parameter `nd` specifies how many rows shall be
processed. This is helpful when parallelizing the matrix-vector product with OpenMP.

The files `sgemv_omp.c` and `sgemv_cmo_omp.c` contain examples for OpenMP parallelization
drivers. They need to be compiler with `ncc` in order to link the optimized OpenMP stack
from the NEC proprietary compiler.


## Prerequisite

The SGEMV-alike functions are written with LLVM intrinsics, therefore you'll need an LLVM
supporting VE to compile them. Here is one example:

Install LLVM-VE-RV-2.2.0 from
https://sx-aurora.com/repos/llvm/x86_64/llvm-ve-rv-2.2.0-2.2.0-1.x86_64.rpm


Set environment for LLVM-VE-RV
```
. llvmvervvars.sh
```

## Build

```
make
```

