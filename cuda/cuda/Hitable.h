#ifndef HITABLE_H
#define HITABLE_H

#include "Ray.h"

class Material;

struct Hit
{
    float t;
    Vec3 p;
    Vec3 normal;
    Material* material;
};

class Hitable
{
public:
    __device__ virtual bool hit(const Ray& r, float min, float max, Hit& hit) const = 0;
};

#endif