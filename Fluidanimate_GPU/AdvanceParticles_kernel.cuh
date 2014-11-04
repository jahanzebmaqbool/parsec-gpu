// Author : Jahanzeb Maqbool
//	    National University of Science and Technology
// Date   : 03-AUG-2010 1600hrs GMT   
/////////////////////////////////////////////////////////

// Advance Particles Device Kernel function.

#include "ParticleDim.h"
#include "ParticleAtt.h"

// Kernel Version-2 : Deals with the memory coalescing as well.
__global__ void AdvanceParticles_kernel (Cell *cells, int *cnumPars, int numCells, float timeStep)  {
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;	
	const int totalThreads = blockDim.x * gridDim.x;

	//No matter how small is execution grid or how large numCells is,
    	//exactly numCells indices will be processed with perfect memory coalescing
        for(int i = tid; i < numCells; i += totalThreads)
        { 
		Cell &cell = cells[i];
		int np = cnumPars[i];
		for(int j = 0; j < np; ++j)
		{
			Vec3 v_half = cell.hv[j] + cell.a[j]*timeStep;
			cell.p[j] += v_half * timeStep;
			cell.v[j] = cell.hv[j] + v_half;
			cell.v[j] *= 0.5f;
			cell.hv[j] = v_half;
		}
	}
}

