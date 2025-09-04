#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

// AltBn128 field element structure (matching ffiasm)
struct FqElement {
    int32_t shortVal;
    uint32_t type;
    uint64_t longVal[4];
};

// AltBn128 G1 point structures (matching ffiasm)
struct G1PointAffine {
    FqElement x;
    FqElement y;
};

struct G1Point {
    FqElement x;
    FqElement y;
    FqElement z;
    FqElement zz;
    FqElement zzz;
};

// AltBn128 field constants (defined in gpu_msm_kernels.cu)
extern __constant__ uint64_t Fq_prime[4];
extern __constant__ uint64_t Fq_np;

// Field arithmetic and point operation functions are defined in each .cu file as needed
// No function declarations here to avoid redeclaration warnings
