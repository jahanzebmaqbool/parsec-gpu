#ifndef _PARTICLEATT_H_
#define _PARTICLEATT_H_

#include "ParticleDim.h"

struct Cell
{
	Vec3 p[16];
	Vec3 hv[16];
	Vec3 v[16];
	Vec3 a[16];
	float density[16];
};

#endif
