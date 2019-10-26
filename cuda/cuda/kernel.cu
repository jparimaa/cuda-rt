#include "Vec3.h"
#include "Ray.h"
#include "HitableList.h"
#include "Sphere.h"

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

__device__ Vec3 color(const Ray& r, Hitable** d_scene)
{
    Hit hit;
    if ((*d_scene)->hit(r, 0.0f, FLT_MAX, hit))
    {
        return 0.5f * Vec3(hit.normal.x() + 1.0f, hit.normal.y() + 1.0f, hit.normal.z() + 1.0f);
    }
    else
    {
        Vec3 normalizedDir = normalize(r.direction());
        float t = 0.5f * (normalizedDir.y() + 1.0f);
        return (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
    }
}

__global__ void render(uint8_t* framebuffer, int width, int height, Vec3 upperLeft, Vec3 horizontal, Vec3 vertical, Vec3 origin, Hitable** d_scene)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i >= width) || (j >= height))
    {
        return;
    }

    float u = static_cast<float>(i) / static_cast<float>(width);
    float v = static_cast<float>(j) / static_cast<float>(height);
    Ray r(origin, upperLeft + u * horizontal - v * vertical);

    Vec3 c = color(r, d_scene);

    int pixelIndex = j * width * 3 + i * 3;
    framebuffer[pixelIndex + 0] = static_cast<uint8_t>(c.r() * 255);
    framebuffer[pixelIndex + 1] = static_cast<uint8_t>(c.g() * 255);
    framebuffer[pixelIndex + 2] = static_cast<uint8_t>(c.b() * 255);
}

__global__ void createScene(Hitable** d_spheres, Hitable** d_scene)
{
    *(d_spheres) = new Sphere(Vec3(0.0f, 0.0f, -1.0f), 0.5f);
    *(d_spheres + 1) = new Sphere(Vec3(0.0f, -100.5f, -1.0f), 100.0f);
    *d_scene = new HitableList(d_spheres, 2);
}

__global__ void freeScene(Hitable** d_spheres, Hitable** d_scene)
{
    delete *(d_spheres);
    delete *(d_spheres + 1);
    delete *d_scene;
}

int main()
{
    Hitable** d_spheres;
    checkCudaErrors(cudaMalloc((void**)&d_spheres, 2 * sizeof(Hitable*)));
    Hitable** d_scene;
    checkCudaErrors(cudaMalloc((void**)&d_scene, sizeof(Hitable*)));
    createScene<<<1, 1>>>(d_spheres, d_scene);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    uint8_t* framebuffer;
    checkCudaErrors(cudaMallocManaged((void**)&framebuffer, c_framebufferSize));

    int tx = 8;
    int ty = 8;

    dim3 blocks(c_width / tx + 1, c_height / ty + 1);
    dim3 threads(tx, ty);

    // clang-format off
    render<<<blocks, threads>>>(framebuffer, c_width, c_height, 
								Vec3(-2.0, 1.0, -1.0), 
								Vec3(4.0, 0.0, 0.0), 
								Vec3(0.0, 2.0, 0.0), 
								Vec3(0.0, 0.0, 0.0),
								d_scene);
    // clang-format on

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::string fileName = "output.png";
    stbi_write_png(fileName.c_str(), c_width, c_height, 3, (void*)framebuffer, 0);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    freeScene<<<1, 1>>>(d_spheres, d_scene);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_spheres));
    checkCudaErrors(cudaFree(d_scene));
    checkCudaErrors(cudaFree(framebuffer));

    std::cout << "OK" << std::endl;
}