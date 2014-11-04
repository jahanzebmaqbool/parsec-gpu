// Author : Jahanzeb Maqbool
//	    National University of Science and Technology


// Rebiuld Grid Device Kernel method.
#include "ParticleDim.h"
#include "ParticleAtt.h"

// Kernel Version 1 - Non coalesced (Each thread operates on one single Cell)

/*
__global__ void RebuildGrid_kernel (Cell *cells, Cell *cells2, int *cnumPars, int *cnumPars2,
			           int numCells, int nx, int ny, int nz, Vec3 domainMin, Vec3 domainMax, Vec3 delta)  {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;	
	
	if (tid < numCells)
	{
		Cell &cell2 = cells2[tid];
		int np = cnumPars2[tid];
		for(int j = 0; j < np; ++j)
		{
			int ci = (int)((cell2.p[j].x - domainMin.x) / delta.x);
			int cj = (int)((cell2.p[j].y - domainMin.y) / delta.y);
			int ck = (int)((cell2.p[j].z - domainMin.z) / delta.z);
	
			if(ci < 0) ci = 0; else if(ci > (nx-1)) ci = nx-1;
			if(cj < 0) cj = 0; else if(cj > (ny-1)) cj = ny-1;
			if(ck < 0) ck = 0; else if(ck > (nz-1)) ck = nz-1;
			int index = (ck*ny + cj)*nx + ci;
			Cell &cell = cells[index];

			int np2 = cnumPars[index];
			cell.p[np2].x = cell2.p[j].x;
			cell.p[np2].y = cell2.p[j].y;
			cell.p[np2].z = cell2.p[j].z;
			cell.hv[np2].x = cell2.hv[j].x;
			cell.hv[np2].y = cell2.hv[j].y;
			cell.hv[np2].z = cell2.hv[j].z;
			cell.v[np2].x = cell2.v[j].x;
			cell.v[np2].y = cell2.v[j].y;
			cell.v[np2].z = cell2.v[j].z;
			++cnumPars[index];
		}
	}
}
*/


// Kernel version 2--memory coalesced (Each thread can operate on different cells in memory coalesced fashion)

__global__ void RebuildGrid_kernel (Cell *cells, Cell *cells2, int *cnumPars, int *cnumPars2,
			           int numCells, int nx, int ny, int nz, Vec3 domainMin, Vec3 domainMax, Vec3 delta)  {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;	
	const int totalThreads = blockDim.x * gridDim.x;


	//No matter how small is execution grid or how large numCells is,
    	//exactly numCells indices will be processed with perfect memory coalescing
        for(int i = tid; i < numCells; i += totalThreads)
        { 
		Cell &cell2 = cells2[tid];
		int np = cnumPars2[tid];
		for(int j = 0; j < np; ++j)
		{
			int ci = (int)((cell2.p[j].x - domainMin.x) / delta.x);
			int cj = (int)((cell2.p[j].y - domainMin.y) / delta.y);
			int ck = (int)((cell2.p[j].z - domainMin.z) / delta.z);
	
			if(ci < 0) ci = 0; else if(ci > (nx-1)) ci = nx-1;
			if(cj < 0) cj = 0; else if(cj > (ny-1)) cj = ny-1;
			if(ck < 0) ck = 0; else if(ck > (nz-1)) ck = nz-1;
			int index = (ck*ny + cj)*nx + ci;
			Cell &cell = cells[index];

			int np2 = cnumPars[index];
			cell.p[np2].x = cell2.p[j].x;
			cell.p[np2].y = cell2.p[j].y;
			cell.p[np2].z = cell2.p[j].z;
			cell.hv[np2].x = cell2.hv[j].x;
			cell.hv[np2].y = cell2.hv[j].y;
			cell.hv[np2].z = cell2.hv[j].z;
			cell.v[np2].x = cell2.v[j].x;
			cell.v[np2].y = cell2.v[j].y;
			cell.v[np2].z = cell2.v[j].z;
			++cnumPars[index];
		}
	}

}	



