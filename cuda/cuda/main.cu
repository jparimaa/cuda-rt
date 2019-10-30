#include "Vec3.h"
#include "Ray.h"
#include "HitableList.h"
#include "Sphere.h"
#include "Camera.h"
#include "Helpers.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <iostream>
#include <chrono>

const int c_width = 800;
const int c_height = 400;
const int c_numPixels = c_width * c_height;
const size_t c_framebufferSize = 3 * c_numPixels * sizeof(uint8_t);
const int c_numRaysPerPixel = 150;

__device__ Vec3 color(const Ray& r, Hitable** d_scene, curandState* localRandomState)
{
    Ray currentRay = r;
    Vec3 currentAttenuation(1.0f, 1.0f, 1.0f);
    int numBounces = 5;
    for (int i = 0; i < numBounces; ++i)
    {
        Hit hit;
        if ((*d_scene)->hit(currentRay, 0.001f, FLT_MAX, hit))
        {
            Ray scattered;
            Vec3 attenuation;
            if (hit.material->scatter(currentRay, hit, attenuation, scattered, localRandomState))
            {
                currentAttenuation *= attenuation;
                currentRay = scattered;
            }
            else
            {
                return Vec3(0.0f, 0.0f, 0.0f);
            }
        }
        else
        {
            Vec3 normalizedDir = normalize(r.direction());
            float t = 0.5f * (normalizedDir.y() + 1.0f);
            Vec3 c = (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
            return currentAttenuation * c;
        }
    }
    return Vec3(0.0f, 0.0f, 0.0f);
}

__global__ void render(uint8_t* framebuffer, int width, int height, int numRays, Camera** d_camera, Hitable** d_scene, curandState* d_randomState)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i >= width) || (j >= height))
    {
        return;
    }

    int randomStatePixelIndex = j * width + i;
    curandState localRandomState = d_randomState[randomStatePixelIndex];
    Vec3 c(0, 0, 0);

    for (int s = 0; s < numRays; ++s)
    {
        float u = static_cast<float>(i + curand_uniform(&localRandomState)) / static_cast<float>(width);
        float v = static_cast<float>(j + curand_uniform(&localRandomState)) / static_cast<float>(height);
        Ray r = (*d_camera)->getRay(u, v);
        c += color(r, d_scene, &localRandomState);
    }

    c /= static_cast<float>(numRays);
    c = squareRoot(c); // gamma fix
    c = clampMax(c, 1.0f);

    int framebufferPixelIndex = j * width * 3 + i * 3;
    framebuffer[framebufferPixelIndex + 0] = static_cast<uint8_t>(c.r() * 255.99f);
    framebuffer[framebufferPixelIndex + 1] = static_cast<uint8_t>(c.g() * 255.99f);
    framebuffer[framebufferPixelIndex + 2] = static_cast<uint8_t>(c.b() * 255.99f);
}

__global__ void initRandomState(int width, int height, curandState* d_randomState)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height))
    {
        return;
    }
    int pixelIndex = j * width + i;
    curand_init(1984, pixelIndex, 0, &d_randomState[pixelIndex]);
}

__global__ void createScene(Hitable** d_spheres, Hitable** d_scene, Camera** d_camera)
{
    // clang-format off
    d_spheres[0] = new Sphere(Vec3( 0.0f,    0.0f, -1.0f),   0.5f, new Lambertian(Vec3(0.8f, 0.3f, 0.3f)));
    d_spheres[1] = new Sphere(Vec3( 0.0f, -100.5f, -1.0f), 100.0f, new Lambertian(Vec3(0.8f, 0.8f, 0.0f)));
    d_spheres[2] = new Sphere(Vec3( 1.0f,    0.0f, -1.0f),   0.5f, new Reflective(Vec3(0.8f, 0.6f, 0.2f), 1.0f));
    d_spheres[3] = new Sphere(Vec3(-1.0f,    0.0f, -1.0f),   0.5f, new Reflective(Vec3(0.8f, 0.8f, 0.8f), 0.3f));
    // clang-format on

    *d_scene = new HitableList(d_spheres, 4);
    *d_camera = new Camera();
}

__global__ void freeScene(Hitable** d_spheres, Hitable** d_scene, Camera** d_camera)
{
    for (int i = 0; i < 4; ++i)
    {
        delete static_cast<Sphere*>(d_spheres[i])->getMaterial();
        delete d_spheres[i];
    }
    delete *d_scene;
    delete *d_camera;
}

int main()
{
    std::cout << "Started..." << std::endl;

    curandState* d_randomState;
    checkCudaErrors(cudaMalloc((void**)&d_randomState, c_numPixels * sizeof(curandState)));

    uint8_t* framebuffer;
    checkCudaErrors(cudaMallocManaged((void**)&framebuffer, c_framebufferSize));

    Hitable** d_spheres;
    checkCudaErrors(cudaMalloc((void**)&d_spheres, 4 * sizeof(Hitable*)));
    Hitable** d_scene;
    checkCudaErrors(cudaMalloc((void**)&d_scene, sizeof(HitableList*)));
    Camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera*)));
    createScene<<<1, 1>>>(d_spheres, d_scene, d_camera);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    int tx = 8;
    int ty = 8;
    dim3 blocks(c_width / tx + 1, c_height / ty + 1);
    dim3 threads(tx, ty);

    std::cout << "Initializing random state...\n";
    {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        initRandomState<<<blocks, threads>>>(c_width, c_height, d_randomState);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Random state initialized (" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms)" << std::endl;
    }

    std::cout << "Rendering...\n";
    {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        render<<<blocks, threads>>>(framebuffer, c_width, c_height, c_numRaysPerPixel, d_camera, d_scene, d_randomState);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Rendering completed (" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms)" << std::endl;
    }

    std::string fileName = "output.png";
    stbi_write_png(fileName.c_str(), c_width, c_height, 3, (void*)framebuffer, 0);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    freeScene<<<1, 1>>>(d_spheres, d_scene, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_scene));
    checkCudaErrors(cudaFree(d_spheres));
    checkCudaErrors(cudaFree(d_randomState));
    checkCudaErrors(cudaFree(framebuffer));

    std::cout << "OK" << std::endl;
    return 0;
}