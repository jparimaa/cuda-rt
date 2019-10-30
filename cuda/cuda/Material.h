#ifndef MATERIAL_H
#define MATERIAL_H

#include "Ray.h"
#include "Hitable.h"
#include "Helpers.h"

class Material
{
public:
    __device__ virtual bool scatter(const Ray& incoming, const Hit& hit, Vec3& attenuation, Ray& outgoing, curandState* localRandomState) const = 0;
};

class Lambertian : public Material
{
public:
    __device__ Lambertian(const Vec3& albedo) :
        m_albedo(albedo)
    {
    }

    __device__ virtual bool scatter(const Ray& incoming, const Hit& hit, Vec3& attenuation, Ray& outgoing, curandState* localRandomState) const
    {
        Vec3 target = hit.p + hit.normal + randomInUnitSphere(localRandomState);
        outgoing = Ray(hit.p, target - hit.p);
        attenuation = m_albedo;
        return true;
    }

private:
    Vec3 m_albedo;
};

class Reflective : public Material
{
public:
    __device__ Reflective(const Vec3& albedo, float fuzziness) :
        m_albedo(albedo),
        m_fuzziness(fuzziness < 1.0f ? fuzziness : 1.0f)
    {
    }

    __device__ virtual bool scatter(const Ray& incoming, const Hit& hit, Vec3& attenuation, Ray& outgoing, curandState* localRandomState) const
    {
        Vec3 reflected = reflect(normalize(incoming.direction()), hit.normal);
        outgoing = Ray(hit.p, reflected + m_fuzziness * randomInUnitSphere(localRandomState));
        attenuation = m_albedo;
        return (dot(outgoing.direction(), hit.normal) > 0.0f);
    }

private:
    Vec3 m_albedo;
    float m_fuzziness;
};

#endif