// Author : Jahanzeb Maqbool
//	    National University of Science and Technology
// Date   : 03-AUG-2010 1530hrs GMT   
/////////////////////////////////////////////////////////

// Process Collisions Device Kernel function.

#include "ParticleDim.h"
#include "ParticleAtt.h"


// Kernel Version-2 : Deals with the memory coalescing as well.
__global__ void ProcessCollisions_kernel (Cell *cells, int *cnumPars, int numCells, Vec3 domainMin, Vec3 domainMax,
					  float timeStep)  {

    	const float parSize = 0.0002f;
        const float epsilon = 1e-10f;
	const float stiffness = 30000.f;
	const float damping = 128.f;
   
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
			Vec3 pos = cell.p[j] + cell.hv[j] * timeStep;

			float diff = parSize - (pos.x - domainMin.x);
			if(diff > epsilon)
				cell.a[j].x += stiffness*diff - damping*cell.v[j].x;

			diff = parSize - (domainMax.x - pos.x);
			if(diff > epsilon)
				cell.a[j].x -= stiffness*diff + damping*cell.v[j].x;

			diff = parSize - (pos.y - domainMin.y);
			if(diff > epsilon)
				cell.a[j].y += stiffness*diff - damping*cell.v[j].y;

			diff = parSize - (domainMax.y - pos.y);
			if(diff > epsilon)
				cell.a[j].y -= stiffness*diff + damping*cell.v[j].y;

			diff = parSize - (pos.z - domainMin.z);
			if(diff > epsilon)
				cell.a[j].z += stiffness*diff - damping*cell.v[j].z;

			diff = parSize - (domainMax.z - pos.z);
			if(diff > epsilon)
				cell.a[j].z -= stiffness*diff + damping*cell.v[j].z;
		}	
	}

}





