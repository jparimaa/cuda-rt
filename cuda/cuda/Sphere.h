#ifndef SPHERE_H
#define SPHERE_H

#include "Hitable.h"
#include "Vec3.h"

#include <cmath>

class Sphere : public Hitable
{
public:
    __device__ Sphere() {}
    __device__ Sphere(Vec3 center, float radius) :
        m_center(center),
        m_radius(radius)
    {
    }

    __device__ virtual bool hit(const Ray& ray, float min, float max, Hit& hit) const;

private:
    Vec3 m_center;
    float m_radius;
};

__device__ bool Sphere::hit(const Ray& ray, float min, float max, Hit& hit) const
{
    Vec3 oc = ray.origin() - m_center;
    float a = dot(ray.direction(), ray.direction());
    float b = dot(oc, ray.direction());
    float c = dot(oc, oc) - m_radius * m_radius;
    float discriminant = b * b - a * c;
    if (discriminant > 0)
    {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < max && temp > min)
        {
            hit.t = temp;
            hit.p = ray.pointAt(hit.t);
            hit.normal = (hit.p - m_center) / m_radius;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < max && temp > min)
        {
            hit.t = temp;
            hit.p = ray.pointAt(hit.t);
            hit.normal = (hit.p - m_center) / m_radius;
            return true;
        }
    }
    return false;
}

#endif