#ifndef RAY_H
#define RAY_H

#include "Vec3.h"

class Ray
{
public:
    __device__ Ray() {}
    __device__ Ray(const Vec3& origin, const Vec3& direction)
    {
        m_origin = origin;
        m_direction = direction;
    }
    __device__ Vec3 origin() const { return m_origin; }
    __device__ Vec3 direction() const { return m_direction; }
    __device__ Vec3 pointAt(float t) const { return m_origin + t * m_direction; }

private:
    Vec3 m_origin;
    Vec3 m_direction;
};

#endif