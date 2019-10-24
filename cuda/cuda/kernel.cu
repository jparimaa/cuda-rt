#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <iostream>

const int c_width = 640;
const int c_height = 480;
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

__global__ void render(uint8_t* framebuffer, int width, int height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i >= width) || (j >= height))
    {
        return;
    }

    int pixelIndex = j * width * 3 + i * 3;
    framebuffer[pixelIndex + 0] = static_cast<uint8_t>(float(i) / width * 255);
    framebuffer[pixelIndex + 1] = static_cast<uint8_t>(float(j) / height * 255);
    framebuffer[pixelIndex + 2] = static_cast<uint8_t>(0.2 * 255);
}

int main()
{
    uint8_t* framebuffer;
    checkCudaErrors(cudaMallocManaged((void**)&framebuffer, c_framebufferSize));

    int tx = 8;
    int ty = 8;

    dim3 blocks(c_width / tx + 1, c_height / ty + 1);
    dim3 threads(tx, ty);
    render<<<blocks, threads>>>(framebuffer, c_width, c_height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::string fileName = "output.png";
    stbi_write_png(fileName.c_str(), c_width, c_height, 3, (void*)framebuffer, 0);

    checkCudaErrors(cudaFree(framebuffer));
}