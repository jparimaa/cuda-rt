#ifndef HITABLE_LIST_H
#define HITABLE_LIST_H

#include "Hitable.h"

class HitableList : public Hitable
{
public:
    __device__ HitableList() {}
    __device__ HitableList(Hitable** hitables, int numHitables) :
        m_hitables(hitables),
        m_numHitables(numHitables)
    {
    }

    __device__ virtual bool hit(const Ray& ray, float min, float max, Hit& hit) const;

private:
    Hitable** m_hitables;
    int m_numHitables;
};

__device__ bool HitableList::hit(const Ray& ray, float min, float max, Hit& hit) const
{
    Hit tempHit;
    bool hitAnything = false;
    float closestSoFar = max;
    for (int i = 0; i < m_numHitables; ++i)
    {
        if (m_hitables[i]->hit(ray, min, closestSoFar, tempHit))
        {
            hitAnything = true;
            closestSoFar = tempHit.t;
            hit = tempHit;
        }
    }
    return hitAnything;
}

#endif