#ifndef HELPERS_H
#define HELPERS_H

#include "Vec3.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define checkCudaErrors(val) checkCuda((val), #val, __FILE__, __LINE__)

void checkCuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(1);
    }
}

#define RANDVEC3 Vec3(curand_uniform(localRandomState), curand_uniform(localRandomState), curand_uniform(localRandomState))

__device__ Vec3 randomInUnitSphere(curandState* localRandomState)
{
    Vec3 p;
    do
    {
        p = 2.0f * RANDVEC3 - Vec3(1.0f, 1.0f, 1.0f);
    } while (p.squaredLength() >= 1.0f);
    return p;
}

__device__ Vec3 reflect(const Vec3& v, const Vec3& n)
{
    return v - 2.0f * dot(v, n) * n;
}

#endif