#include "Vec3.h"
#include "Ray.h"
#include "HitableList.h"
#include "Sphere.h"
#include "Camera.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <iostream>

const int c_width = 800;
const int c_height = 400;
const int c_numPixels = c_width * c_height;
const size_t c_framebufferSize = 3 * c_numPixels * sizeof(uint8_t);
const int c_numRaysPerPixel = 1;

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

__device__ Vec3 color(const Ray& r, Hitable** d_scene /*, curandState* localRandomState*/)
{
    Ray currentRay = r;
    float currentAttenuation = 1.0f;
    int numBounces = 50;
    for (int i = 0; i < numBounces; ++i)
    {
        Hit hit;
        if ((*d_scene)->hit(currentRay, 0.001f, FLT_MAX, hit))
        {
            Vec3 target = hit.p + hit.normal /* + randomInUnitSphere(localRandomState)*/;
            currentAttenuation *= 0.5f;
            currentRay = Ray(hit.p, target - hit.p);
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
    int pixelIndex = j * width * 3 + i * 3;

    //curandState localRandomState = d_randomState[pixelIndex];
    Vec3 c(0, 0, 0);

    for (int s = 0; s < numRays; ++s)
    {
        float u = static_cast<float>(i /*+ curand_uniform(&localRandomState)*/) / static_cast<float>(width);
        float v = static_cast<float>(j /*+ curand_uniform(&localRandomState)*/) / static_cast<float>(height);
        Ray r = (*d_camera)->getRay(u, v);
        c += color(r, d_scene /*, &localRandomState*/);
    }

    //d_randomState[pixelIndex] = localRandomState;

    framebuffer[pixelIndex + 0] = static_cast<uint8_t>(c.r() * 255);
    framebuffer[pixelIndex + 1] = static_cast<uint8_t>(c.g() * 255);
    framebuffer[pixelIndex + 2] = static_cast<uint8_t>(c.b() * 255);
}

__global__ void initRender(int width, int height, curandState* randomState)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height))
    {
        return;
    }
    int pixelIndex = j * width + i;
    curand_init(1984, pixelIndex, 0, &randomState[pixelIndex]);
}

__global__ void createScene(Hitable** d_spheres, Hitable** d_scene, Camera** d_camera)
{
    *(d_spheres) = new Sphere(Vec3(0.0f, 0.0f, -1.0f), 0.5f);
    *(d_spheres + 1) = new Sphere(Vec3(0.0f, -100.5f, -1.0f), 100.0f);
    *d_scene = new HitableList(d_spheres, 2);
    *d_camera = new Camera();
}

__global__ void freeScene(Hitable** d_spheres, Hitable** d_scene, Camera** d_camera)
{
    delete *(d_spheres);
    delete *(d_spheres + 1);
    delete *d_scene;
    delete *d_camera;
}

int main()
{
    curandState* d_randomState;
    checkCudaErrors(cudaMalloc((void**)&d_randomState, c_numPixels * sizeof(curandState)));

    uint8_t* framebuffer;
    checkCudaErrors(cudaMallocManaged((void**)&framebuffer, c_framebufferSize));

    Hitable** d_spheres;
    checkCudaErrors(cudaMalloc((void**)&d_spheres, 2 * sizeof(Hitable*)));
    Hitable** d_scene;
    checkCudaErrors(cudaMalloc((void**)&d_scene, sizeof(Hitable*)));
    Camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera*)));
    createScene<<<1, 1>>>(d_spheres, d_scene, d_camera);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    int tx = 8;
    int ty = 8;
    dim3 blocks(c_width / tx + 1, c_height / ty + 1);
    dim3 threads(tx, ty);

    //initRender<<<blocks, threads>>>(c_width, c_height, d_randomState);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(framebuffer, c_width, c_height, c_numRaysPerPixel, d_camera, d_scene, d_randomState);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::string fileName = "output.png";
    stbi_write_png(fileName.c_str(), c_width, c_height, 3, (void*)framebuffer, 0);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    freeScene<<<1, 1>>>(d_spheres, d_scene, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_scene));
    checkCudaErrors(cudaFree(d_spheres));
    //checkCudaErrors(cudaFree(d_randomState));
    checkCudaErrors(cudaFree(framebuffer));

    std::cout << "OK" << std::endl;
}