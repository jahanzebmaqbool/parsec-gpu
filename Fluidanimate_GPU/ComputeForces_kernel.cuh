// Author : Jahanzeb Maqbool
//	    National University of Science and Technology


// Compute Forces Device Kernel Function.

#include "ParticleDim.h"
#include "ParticleAtt.h"


// inline device functions.

/*
__device__ float deviceMax (float a, b) {
	return ( a > b ) ? a : b; 
}	
__device float deviceSqrt (float x) {
	
}
*/
__device__ float   GetLengthSq(Vec3 p) {
	return p.x*p.x + p.y*p.y + p.z*p.z; 
}
__device__ float   GetLength(Vec3 p) {
    return sqrtf(GetLengthSq(p)); 
}
//__device__ Vec3 &  Normalize() {
//	 return *this /= GetLength(); 
//}

//////////////////////////////////////////////////////////////////////

/*
    //Vec3 &  operator += (Vec3 const &v) { x += v.x;  y += v.y; z += v.z; return *this; }
    Vec3 PlusEqual	
    Vec3 &  operator -= (Vec3 const &v) { x -= v.x;  y -= v.y; z -= v.z; return *this; }
    Vec3 &  operator *= (float s)       { x *= s;  y *= s; z *= s; return *this; }
    Vec3 &  operator /= (float s)       { x /= s;  y /= s; z /= s; return *this; }

    Vec3    operator + (Vec3 const &v) const    { return Vec3(x+v.x, y+v.y, z+v.z); }
    Vec3    operator - () const                 { return Vec3(-x, -y, -z); }
    Vec3    operator - (Vec3 const &v) const    { return Vec3(x-v.x, y-v.y, z-v.z); }
    Vec3    operator * (float s) const          { return Vec3(x*s, y*s, z*s); }
    Vec3    operator / (float s) const          { return Vec3(x/s, y/s, z/s); }	
    float   operator * (Vec3 const &v) const    { return x*v.x + y*v.y + z*v.z; }
*/

//////////////////////////////////////////////////////////////////////
// function: called by kernel to find the neighbours of a cell

__device__ int GetNeighborCells(int ci, int cj, int ck, int *neighCells, int nx, int ny, int nz, int *cnumPars)
{
	int numNeighCells = 0;

	for(int di = -1; di <= 1; ++di)
		for(int dj = -1; dj <= 1; ++dj)
			for(int dk = -1; dk <= 1; ++dk)
			{
				int ii = ci + di;
				int jj = cj + dj;
				int kk = ck + dk;
				if(ii >= 0 && ii < nx && jj >= 0 && jj < ny && kk >= 0 && kk < nz)
				{
					int index = (kk*ny + jj)*nx + ii;
					if(cnumPars[index] != 0)
					{
						neighCells[numNeighCells] = index;
						++numNeighCells;
					}
				}
			}

	return numNeighCells;
}



// Data is already present in GPU memory from previous step of "Rebuild Grid".
// Just passing the pointers to the data.

__global__ void ComputeForces_kernel (Cell *cells, int *cnumPars, 
				      int numCells, int nx, int ny, int nz, Vec3 domainMin, Vec3 domainMax, Vec3 delta,
				      // exclusive parameters to ComputeForces.....	
				      Vec3 externalAcceleration, float hSq, float densityCoeff, float h,
				      float pressureCoeff, float doubleRestDensity, float viscosityCoeff) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;	
	const int totalThreads = blockDim.x * gridDim.x;

	for(int i = tid; i < numCells; i += totalThreads)
        { 
	 	Cell &cell = cells[i];
		int np = cnumPars[i];
		for(int j = 0; j < np; ++j)
		{
			cell.density[j] = 0.f;
			cell.a[j] = externalAcceleration;
		}
	}    	
	__syncthreads ();
	
	int neighCells[27];

	// going to find neighbour cells.
	// converting 1-D index to 3-D indices (i,j,k)


//!!!!!! problem seems here..... because tid is uniqe in each thread block, so in multiple thread blocks tid gets repeating... so (ci, cj, ck) also gets repeated..



	for(int i = tid; i < numCells; i += totalThreads)
    	{

		int ci, cj, ck;
		ci = i % nx;
		cj = (i%(nx*ny))/nx ;	
		ck = i/(nx*ny);	
		

		int numPars = cnumPars [i];   		
		if(numPars == 0)
         		continue;
		
		int numNeighCells = GetNeighborCells(ci, cj, ck, neighCells, nx, ny, nz, cnumPars);
		Cell &cell = cells[i];

		// for each particle of the 'cell' find the neighbor cells and the neighbor particles in that cells.
		for(int ipar = 0; ipar < numPars; ++ipar) // numPars --> Max limit = 16
		    for(int inc = 0; inc < numNeighCells; ++inc)  // numNeighCells --> Max limit = 27
      		    {
                	int cindexNeigh = neighCells[inc];
				Cell &neigh = cells[cindexNeigh];
		        int numNeighPars = cnumPars[cindexNeigh];

		        for(int iparNeigh = 0; iparNeigh < numNeighPars; ++iparNeigh)
					if(&neigh.p[iparNeigh] < &cell.p[ipar])
					{
					  //float distSq = (cell.p[ipar] - neigh.p[iparNeigh]).GetLengthSq();			      
				   
					 // Vec3 distSqVec = (cell.p[ipar] - neigh.p[iparNeigh]);	
					 // float distSq1 = GetLengthSq (distSqVec);	 		      
		 		          
					  float distSq = GetLengthSq ( (cell.p[ipar] - neigh.p[iparNeigh]) );
		

					  if(distSq < hSq)
					  {
						float t = hSq - distSq;
						float tc = t*t*t;
		       /*
			   if(border[i]){
			   do
			   {
			   while(lock[i]){}// busy waiting
				lock[i] = true;
			   }while(!lock[i])
			   
			   }
			   
			   */
						cell.density[ipar] += tc;
						neigh.density[iparNeigh] += tc;
					  }
					}
	
	           }      
	}
	// synchronize all threads of a block. -- Stage 01
	__syncthreads ();
	
	const float tc = hSq*hSq*hSq;
	for(int i = tid; i < numCells; i += totalThreads)
    	{
		Cell &cell = cells[i];
		int np = cnumPars[i];
		for(int j = 0; j < np; ++j)
		{
			cell.density[j] += tc;
			cell.density[j] *= densityCoeff;
		}
	}
	// synchronize all threads of a block. -- Stage 02
	//__syncthreads ();
		
	
	for(int i = tid; i < numCells; i += totalThreads)
    	{ 
               int ci, cj, ck;
                ci = i % nx;
                cj = (i%(nx*ny))/nx ;
                ck = i/(nx*ny);

		int numPars = cnumPars [i];
		if(numPars == 0)
			continue;
		int numNeighCells = GetNeighborCells(ci, cj, ck, neighCells, nx, ny, nz, cnumPars);

		Cell &cell = cells[i];
		for(int ipar = 0; ipar < numPars; ++ipar)
			for(int inc = 0; inc < numNeighCells; ++inc)
			{
				int cindexNeigh = neighCells[inc];
				Cell &neigh = cells[cindexNeigh];
				int numNeighPars = cnumPars[cindexNeigh];
				for(int iparNeigh = 0; iparNeigh < numNeighPars; ++iparNeigh)
					if(&neigh.p[iparNeigh] < &cell.p[ipar])
					{
						Vec3 disp = cell.p[ipar] - neigh.p[iparNeigh];
						
						float distSq = GetLengthSq(disp);
						if(distSq < hSq)
						{
							//float dist = sqrtf(std::max(distSq, 1e-12f));
							//float dist = deviceSqrt (deviceMax (distSq, 1e-12f));
							float dist = sqrtf (fmaxf(distSq, 1e-12f));
							
							float hmr = h - dist;

							Vec3 acc = disp * pressureCoeff * (hmr*hmr/dist) * (cell.density[ipar]+neigh.density[iparNeigh] - doubleRestDensity);

	/*if (tid == 33)
	{
		printf ("disp %f \n", disp.x);
		printf ("press %f \n", pressureCoeff);
		printf ("hmr %f \n", (hmr*hmr/dist));
		printf ("zzzzz %f \n", (cell.density[ipar]+neigh.density[iparNeigh]));
		printf ("drd %f \n", doubleRestDensity);
	
		printf ("acc %f \n", (disp * pressureCoeff * (hmr*hmr/dist) * (cell.density[ipar]+neigh.density[iparNeigh] - doubleRestDensity)).x  );
	
	}*/
							acc += (neigh.v[iparNeigh] - cell.v[ipar]) * viscosityCoeff * hmr;
							acc /= cell.density[ipar] * neigh.density[iparNeigh];

							cell.a[ipar] += acc;
							neigh.a[iparNeigh] -= acc;
						}
					}
			}
	
	}
	
} // kernel ends...












	
	
		
	
