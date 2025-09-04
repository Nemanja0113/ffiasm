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

// Field arithmetic function declarations (defined in gpu_msm_kernels.cu)
// Note: These are declared without __device__ here since they're defined in a different .cu file
bool fq_is_zero(const FqElement* a);
void fq_zero(FqElement* result);
void fq_one(FqElement* result);
void fq_copy(FqElement* result, const FqElement* a);
void fq_add(FqElement* result, const FqElement* a, const FqElement* b);
void fq_sub(FqElement* result, const FqElement* a, const FqElement* b);
void fq_mul(FqElement* result, const FqElement* a, const FqElement* b);

// Point operation function declarations (defined in gpu_msm_kernels.cu)
// Note: These are declared without __device__ here since they're defined in a different .cu file
bool point_is_zero(const G1Point* a);
void point_zero(G1Point* result);
void point_copy(G1Point* result, const G1Point* src);
void point_add(G1Point* result, const G1Point* a, const G1Point* b);
void point_add(G1Point* result, const G1Point* a, const G1PointAffine* b);
void point_sub(G1Point* result, const G1Point* a, const G1PointAffine* b);
