#include "Vec3.h"
#include "Ray.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <iostream>

const int c_width = 800;
const int c_height = 400;
const int c_numPixels = c_width * c_height;
const size_t c_framebufferSize = 3 * c_numPixels * sizeof(uint8_t);

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

__device__ bool hitSphere(const Vec3& center, float radius, const Ray& r)
{
    Vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0f * dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4.0f * a * c;
    return discriminant > 0.0f;
}

__device__ Vec3 color(const Ray& r)
{
    if (hitSphere(Vec3(0.0f, 0.0f, -1.0f), 0.5f, r))
    {
        return Vec3(1.0f, 0.0f, 0.0f);
    }

    Vec3 dir = normalize(r.direction());
    float t = 0.5f * (dir.y() + 1.0f);
    return (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
}

__global__ void render(uint8_t* framebuffer, int width, int height, Vec3 lowerLeft, Vec3 horizontal, Vec3 vertical, Vec3 origin)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i >= width) || (j >= height))
    {
        return;
    }

    int pixelIndex = j * width * 3 + i * 3;

    float u = static_cast<float>(i) / static_cast<float>(width);
    float v = static_cast<float>(j) / static_cast<float>(height);
    Ray r(origin, lowerLeft + u * horizontal + v * vertical);

    Vec3 c = color(r);

    framebuffer[pixelIndex + 0] = static_cast<uint8_t>(c.r() * 255);
    framebuffer[pixelIndex + 1] = static_cast<uint8_t>(c.g() * 255);
    framebuffer[pixelIndex + 2] = static_cast<uint8_t>(c.b() * 255);
}

int main()
{
    uint8_t* framebuffer;
    checkCudaErrors(cudaMallocManaged((void**)&framebuffer, c_framebufferSize));

    int tx = 8;
    int ty = 8;

    dim3 blocks(c_width / tx + 1, c_height / ty + 1);
    dim3 threads(tx, ty);
    // clang-format off
    render<<<blocks, threads>>>(framebuffer, c_width, c_height, 
								Vec3(-2.0, -1.0, -1.0), 
								Vec3(4.0, 0.0, 0.0), 
								Vec3(0.0, 2.0, 0.0), 
								Vec3(0.0, 0.0, 0.0));
    // clang-format on
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::string fileName = "output.png";
    stbi_write_png(fileName.c_str(), c_width, c_height, 3, (void*)framebuffer, 0);

    checkCudaErrors(cudaFree(framebuffer));

    std::cout << "OK" << std::endl;
}