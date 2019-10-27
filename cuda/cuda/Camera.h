#ifndef CAMERA_H
#define CAMERA_H

#include "Ray.h"

class Camera
{
public:
    __device__ Camera() :
        m_upperLeftCorner(-2.0f, 1.0f, -1.0f),
        m_horizontal(4.0f, 0.0f, 0.0f),
        m_vertical(0.0f, 2.0f, 0.0f),
        m_origin(0.0f, 0.0f, 0.0f)
    {
    }
    __device__ Ray getRay(float u, float v)
    {
        return Ray(m_origin, m_upperLeftCorner + u * m_horizontal - v * m_vertical - m_origin);
    }

private:
    Vec3 m_origin;
    Vec3 m_upperLeftCorner;
    Vec3 m_horizontal;
    Vec3 m_vertical;
};

#endif