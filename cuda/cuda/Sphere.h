#ifndef SPHERE_H
#define SPHERE_H

#include "Hitable.h"
#include "Vec3.h"
#include "Material.h"

class Sphere : public Hitable
{
public:
    __device__ Sphere() {}
    __device__ Sphere(Vec3 center, float radius, Material* material) :
        m_center(center),
        m_radius(radius),
        m_material(material)
    {
    }

    __device__ virtual bool hit(const Ray& ray, float min, float max, Hit& hit) const
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
                hit.material = m_material;
                return true;
            }
            temp = (-b + sqrt(discriminant)) / a;
            if (temp < max && temp > min)
            {
                hit.t = temp;
                hit.p = ray.pointAt(hit.t);
                hit.normal = (hit.p - m_center) / m_radius;
                hit.material = m_material;
                return true;
            }
        }
        return false;
    }

    __device__ Material* getMaterial() const { return m_material; }

private:
    Vec3 m_center;
    float m_radius;
    Material* m_material;
};

#endif