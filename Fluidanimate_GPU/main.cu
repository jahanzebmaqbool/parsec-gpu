/* 
*  Application   : PARSEC benchmark fluid animate GPU Version.
*  Serial Code   : PARSEC benchmark fluid animate
*  Serial Author : Code originally written by Richard O. Lee 
*                  Modified by Christian Bienia and Christian Fensch
*
*  GPU Version Author : Jahanzeb Maqbool 
*                       National University of Science and Technology
*
*/
/////////////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <fstream>
#include <assert.h>
#include <string.h>
#include <cutil_inline.h>

// Class Representing Particle Dimenstions in space.
#include "ParticleDim.h"
// Struct representing Particle Attributes.
#include "ParticleAtt.h"
// Rebuild Grid Kernel.
#include "RebuildGrid_kernel.cuh"
// Compute Forces Kernel.
#include "ComputeForces_kernel.cuh"
// Process Collisions kernel.
#include "ProcessCollisions_kernel.cuh"
// Advance Particles kernel.
#include "AdvanceParticles_kernel.cuh"

void RebuildGrid ();


#define DEBUG false
////////////////////////////////////////////////////////////////////////////////

	/**
	 * C++ version 0.4 char* style "itoa":
	 * Written by Lukás Chmela
	 * Released under GPLv3.
	 */
	char* itoa(int value, char* result, int base) {
		// check that the base if valid
		if (base < 2 || base > 36) { *result = '\0'; return result; }
	
		char* ptr = result, *ptr1 = result, tmp_char;
		int tmp_value;
	
		do {
			tmp_value = value;
			value /= base;
			*ptr++ = "zyxwvutsrqponmlkjihgfedcba9876543210123456789abcdefghijklmnopqrstuvwxyz" [35 + (tmp_value - value * base)];
		} while ( value );
	
		// Apply negative sign
		if (tmp_value < 0) *ptr++ = '-';
		*ptr-- = '\0';
		while(ptr1 < ptr) {
			tmp_char = *ptr;
			*ptr--= *ptr1;
			*ptr1++ = tmp_char;
		}
		return result;
	}



////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

static inline int isLittleEndian() {
  union {
    uint16_t word;
    uint8_t byte;
  } endian_test;

  endian_test.word = 0x00FF;
  return (endian_test.byte == 0xFF);
}

union __float_and_int {
  uint32_t i;
  float    f;
};


static inline float bswap_float(float x) {
  union __float_and_int __x;

   __x.f = x;
   __x.i = ((__x.i & 0xff000000) >> 24) | ((__x.i & 0x00ff0000) >>  8) |
           ((__x.i & 0x0000ff00) <<  8) | ((__x.i & 0x000000ff) << 24);

  return __x.f;
}

static inline int bswap_int32(int x) {
  return ( (((x) & 0xff000000) >> 24) | (((x) & 0x00ff0000) >>  8) |
           (((x) & 0x0000ff00) <<  8) | (((x) & 0x000000ff) << 24) );
}

////////////////////////////////////////////////////////////////////////////////

const float timeStep = 0.005f;
const float doubleRestDensity = 2000.f;
const float kernelRadiusMultiplier = 1.695f;
const float stiffness = 1.5f;
const float viscosity = 0.4f;
const Vec3 externalAcceleration(0.f, -9.8f, 0.f);
const Vec3 domainMin(-0.065f, -0.08f, -0.065f);
const Vec3 domainMax(0.065f, 0.1f, 0.065f);

float restParticlesPerMeter, h, hSq;
float densityCoeff, pressureCoeff, viscosityCoeff;

int nx, ny, nz;     // number of grid cells in each dimension
Vec3 delta;         // cell dimensions
int origNumParticles = 0;
int numParticles = 0;
int numCells = 0;
Cell *cells = 0;
Cell *cells2 = 0;
int *cnumPars = 0;
int *cnumPars2 = 0;

Cell *d_cells = 0;
Cell *d_cells2 = 0;
int  *d_cnumPars = 0;
int  *d_cnumPars2 = 0;


////////////////////////////////////////////////////////////////////////////////

void InitSim(char const *fileName)
{
	std::cout << "Loading file \"" << fileName << "\"..." << std::endl;
	std::ifstream file(fileName, std::ios::binary);
	assert(file);

	file.read((char *)&restParticlesPerMeter, 4);
	file.read((char *)&origNumParticles, 4);
        if(!isLittleEndian()) {
          restParticlesPerMeter = bswap_float(restParticlesPerMeter);
          origNumParticles      = bswap_int32(origNumParticles);
        }
	numParticles = origNumParticles;

	h = kernelRadiusMultiplier / restParticlesPerMeter;
	hSq = h*h;
	const float pi = 3.14159265358979f;
	float coeff1 = 315.f / (64.f*pi*pow(h,9.f));
	float coeff2 = 15.f / (pi*pow(h,6.f));
	float coeff3 = 45.f / (pi*pow(h,6.f));
	float particleMass = 0.5f*doubleRestDensity / (restParticlesPerMeter*restParticlesPerMeter*restParticlesPerMeter);
	densityCoeff = particleMass * coeff1;
	pressureCoeff = 3.f*coeff2 * 0.5f*stiffness * particleMass;



//	printf ("===============================asdasd1 %f \n", pow(h,9.f) );			
//	printf ("===============================coff2 %f \n", coeff3 );
//	printf ("===============================stff %f \n", stiffness );
//	printf ("===============================mass %f \n", particleMass );
//	printf ("===============================cu press coff %f \n", pressureCoeff );
	
	viscosityCoeff = viscosity * coeff3 * particleMass;

	Vec3 range = domainMax - domainMin;
	nx = (int)(range.x / h);
	ny = (int)(range.y / h);
	nz = (int)(range.z / h);

	if (DEBUG)
	printf ("nx : %d  , ny : %d  ,  nz  %d \n", nx, ny, nz);	
	
	assert(nx >= 1 && ny >= 1 && nz >= 1);
	numCells = nx*ny*nz;
	if (DEBUG)
	std::cout << "Number of cells: " << numCells << std::endl;
	delta.x = range.x / nx;
	delta.y = range.y / ny;
	delta.z = range.z / nz;
	assert(delta.x >= h && delta.y >= h && delta.z >= h);


        // allocating cpu memory... 
	cells = (Cell*) malloc(numCells*sizeof(Cell));	
	cnumPars = (int*) malloc(numCells*sizeof(int));	
	cells2 = (Cell*) malloc(numCells*sizeof(Cell));	
	cnumPars2 = (int*) malloc(numCells*sizeof(int));		

        // allocating gpu memory...
    	cutilSafeCall( cudaMalloc((void **)&d_cells,  (numCells*sizeof(Cell)) ) );
    	cutilSafeCall( cudaMalloc((void **)&d_cells2,  (numCells*sizeof(Cell)) ) );
	cutilSafeCall( cudaMalloc((void **)&d_cnumPars,  (numCells*sizeof(int)) ) );
	cutilSafeCall( cudaMalloc((void **)&d_cnumPars2,  (numCells*sizeof(int)) ) );

	assert(cells && cells2 && cnumPars && cnumPars2 && d_cells && d_cells2 && d_cnumPars && d_cnumPars2 );
	memset(cnumPars2, 0, numCells*sizeof(int));

	if (DEBUG)
        printf ("Mem init Done....\n"); 

	float px, py, pz, hvx, hvy, hvz, vx, vy, vz;
	for(int i = 0; i < origNumParticles; ++i)
	{
		file.read((char *)&px, 4);
		file.read((char *)&py, 4);
		file.read((char *)&pz, 4);
		file.read((char *)&hvx, 4);
		file.read((char *)&hvy, 4);
		file.read((char *)&hvz, 4);
		file.read((char *)&vx, 4);
		file.read((char *)&vy, 4);
		file.read((char *)&vz, 4);
                if(!isLittleEndian()) {
                  px  = bswap_float(px);
                  py  = bswap_float(py);
                  pz  = bswap_float(pz);
                  hvx = bswap_float(hvx);
                  hvy = bswap_float(hvy);
                  hvz = bswap_float(hvz);
                  vx  = bswap_float(vx);
                  vy  = bswap_float(vy);
                  vz  = bswap_float(vz);
                }

		int ci = (int)((px - domainMin.x) / delta.x);
		int cj = (int)((py - domainMin.y) / delta.y);
		int ck = (int)((pz - domainMin.z) / delta.z);

		if(ci < 0) ci = 0; else if(ci > (nx-1)) ci = nx-1;
		if(cj < 0) cj = 0; else if(cj > (ny-1)) cj = ny-1;
		if(ck < 0) ck = 0; else if(ck > (nz-1)) ck = nz-1;

		int index = (ck*ny + cj)*nx + ci;
		Cell &cell = cells2[index];

		int np = cnumPars2[index];
		if(np < 16)
		{
			cell.p[np].x = px;
			cell.p[np].y = py;
			cell.p[np].z = pz;
			cell.hv[np].x = hvx;
			cell.hv[np].y = hvy;
			cell.hv[np].z = hvz;
			cell.v[np].x = vx;
			cell.v[np].y = vy;
			cell.v[np].z = vz;
			++cnumPars2[index];
		}
		else
			--numParticles;
	}

	if (DEBUG)
	std::cout << "Number of particles: " << numParticles << " (" << origNumParticles-numParticles << " skipped)" << std::endl;
	
	
	RebuildGrid ();
}


void CopyMemHtoD ()
{
    // copying required data from CPU memory to GPU memory...
    cutilSafeCall( cudaMemcpy(d_cells,  cells,   numCells*sizeof(Cell), cudaMemcpyHostToDevice) );   
    cutilSafeCall( cudaMemcpy(d_cells2,  cells2,   numCells*sizeof(Cell), cudaMemcpyHostToDevice) );   
    cutilSafeCall( cudaMemcpy(d_cnumPars,  cnumPars,   numCells*sizeof(int), cudaMemcpyHostToDevice) );
    cutilSafeCall( cudaMemcpy(d_cnumPars2,  cnumPars2,   numCells*sizeof(int), cudaMemcpyHostToDevice) );   
}

void CopyMemDtoH ()
{
    // copy back memory from GPU to CPU.
    cutilSafeCall( cudaMemcpy(cells, d_cells, numCells*sizeof(Cell), cudaMemcpyDeviceToHost) );     
    cutilSafeCall( cudaMemcpy(cells2, d_cells2, numCells*sizeof(Cell), cudaMemcpyDeviceToHost) );     
    cutilSafeCall( cudaMemcpy(cnumPars, d_cnumPars, numCells*sizeof(int), cudaMemcpyDeviceToHost) ); 
    cutilSafeCall( cudaMemcpy(cnumPars2, d_cnumPars2, numCells*sizeof(int), cudaMemcpyDeviceToHost) );        
}

////////////////////////////////////////////////////////////////////////////////

void SaveFile(char const *fileName)
{
	std::cout << "Saving file \"" << fileName << "\"..." << std::endl;

	std::ofstream file(fileName, std::ios::binary);
	assert(file);

        if(!isLittleEndian()) {
          float restParticlesPerMeter_le;
          int   origNumParticles_le;

          restParticlesPerMeter_le = bswap_float(restParticlesPerMeter);
          origNumParticles_le      = bswap_int32(origNumParticles);
	  file.write((char *)&restParticlesPerMeter_le, 4);
	  file.write((char *)&origNumParticles_le,      4);
        } else {
	  file.write((char *)&restParticlesPerMeter, 4);
	  file.write((char *)&origNumParticles, 4);
        }

	if (DEBUG)
	printf ("num cells %d , cell[15].p[0].x %f \n", numCells, cells[15].p[5].x);

	int count = 0;

	for(int i = 0; i < numCells; ++i)
	{
		Cell &cell = cells[i];
		int np = cnumPars[i];

		//printf (" ==================== np %d\n", np);

		for(int j = 0; j < np; ++j)
		{
                        if(!isLittleEndian()) {
                          float px, py, pz, hvx, hvy, hvz, vx,vy, vz;


		//		printf ("=================big indian\n");
                          px  = bswap_float(cell.p[j].x);
                          py  = bswap_float(cell.p[j].y);
                          pz  = bswap_float(cell.p[j].z);
                          hvx = bswap_float(cell.hv[j].x);
                          hvy = bswap_float(cell.hv[j].y);
                          hvz = bswap_float(cell.hv[j].z);
                          vx  = bswap_float(cell.v[j].x);
                          vy  = bswap_float(cell.v[j].y);
                          vz  = bswap_float(cell.v[j].z);

			  file.write((char *)&px,  4);
			  file.write((char *)&py,  4);
			  file.write((char *)&pz,  4);
			  file.write((char *)&hvx, 4);
			  file.write((char *)&hvy, 4);
			  file.write((char *)&hvz, 4);
			  file.write((char *)&vx,  4);
			  file.write((char *)&vy,  4);
			  file.write((char *)&vz,  4);
                        } else {

				//printf ("====================little indian\n");

			  file.write((char *)&cell.p[j].x,  4);
			  file.write((char *)&cell.p[j].y,  4);
			  file.write((char *)&cell.p[j].z,  4);
			  file.write((char *)&cell.hv[j].x, 4);
			  file.write((char *)&cell.hv[j].y, 4);
			  file.write((char *)&cell.hv[j].z, 4);
			  file.write((char *)&cell.v[j].x,  4);
			  file.write((char *)&cell.v[j].y,  4);
			  file.write((char *)&cell.v[j].z,  4);
                        }
			++count;
		}
	}

	//printf ("count : %d   and   No. Particles : %d \n", count, numParticles);
	assert(count == numParticles);

	int numSkipped = origNumParticles - numParticles;
	float zero = 0.f;
        if(!isLittleEndian()) {
          zero = bswap_float(zero);
        }
	for(int i = 0; i < numSkipped; ++i)
	{
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
	}
}

////////////////////////////////////////////////////////////////////////////////

void RebuildGrid()
{
	memset(cnumPars, 0, numCells*sizeof(int));

	for(int i = 0; i < numCells; ++i)
	{
		Cell const &cell2 = cells2[i];
		int np = cnumPars2[i];
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

////////////////////////////////////////////////////////////////////////////////

void MemFree()
{
	free (cells);
    	free (cells2);
	free (cnumPars);
	free (cnumPars2);
	
	cutilSafeCall( cudaFree(d_cells)  );
	cutilSafeCall( cudaFree(d_cells2)  );
	cutilSafeCall( cudaFree(d_cnumPars)  );
	cutilSafeCall( cudaFree(d_cnumPars2)  );
}


////////////////////////////////////////////////////////////////////////////////
/*
int ValidateResult (char *gfileName, char* cfileName, double &sqrSum) 
{
	
	std::cout << "Opening gpu result file \"" << gfileName << "\"..." << std::endl;
	std::cout << "Opening cpu result file \"" << cfileName << "\"..." << std::endl;

	std::ifstream gfile(gfileName, std::ios::binary);
	std::ifstream cfile(cfileName, std::ios::binary);
	
	assert(cfile && gfile);
	
	float cTemp, gTemp;
	int errCount = 0;
	int gsize = 0;
	int csize = 0;
	
	gfile.seekg(0,std::ios::end);
	gsize = gfile.tellg();	
	gsize /= 4;
//	gsize /= 1024;
	gfile.seekg (0, std::ios::beg);

	cfile.seekg(0,std::ios::end);
	csize = cfile.tellg();
	csize /= 4;	
//	csize /=1024;	
	cfile.seekg (0, std::ios::beg);

	assert (gsize == csize);
	
	printf ("gsize = %d , csize = %d \n", gsize, csize);

	for (int i = 0; i < 20 ; i++)
	{
		gfile.read((char *)&cTemp,  4);
		cfile.read((char *)&gTemp, 4);

		if(!isLittleEndian()) {
	          cTemp = bswap_float(cTemp);
		  gTemp = bswap_float(gTemp);	
		}
		

		printf ("gFile : %f , cFile : %f  \n", gTemp, cTemp);
		if (cTemp != gTemp)
		    errCount ++;
				
		//sqrSum += ( (cTemp - gTemp)*(cTemp - gTemp) );
		//printf ("in function sqr sum : %f\n", sqrSum);
		
	}

	return errCount;
}
*/
////////////////////////////////////////////////////////////////////////////////

void AdvanceFrame_GPU (int numBlocks, int threads_per_block, int iter)
{
    
/*	if (iter == 0) {
	
    // calling phase-1 Rebuild Grid kernel...  
    RebuildGrid_kernel <<< numBlocks, threads_per_block >>> (d_cells, d_cells2, d_cnumPars, d_cnumPars2 , numCells,
							     nx, ny, nz, domainMin, domainMax, delta);	
							    //? where did you copied numCells,nx,ny,nz,domainMax,domainMin,Delta.... = solved= auto copy in CUDA
								
	}
*/	
    // calling phase-2 Compute Forces Kernel...
    ComputeForces_kernel <<< numBlocks, threads_per_block >>> (d_cells, d_cnumPars, numCells,
	//ComputeForces_kernel <<< numBlocks, threads_per_block >>> (cells, cnumPars, numCells,
							     nx, ny, nz, domainMin, domainMax, delta, externalAcceleration, hSq, densityCoeff, h, pressureCoeff,
								 doubleRestDensity, viscosityCoeff );
							    // same issue as above... = solved as above

    // calling phase-3 Process Collision kernel...
    ProcessCollisions_kernel <<< numBlocks, threads_per_block >>> (d_cells, d_cnumPars, numCells, domainMin, domainMax, timeStep);	
	//ProcessCollisions_kernel <<< numBlocks, threads_per_block >>> (cells, cnumPars, numCells, domainMin, domainMax, timeStep);	
							   // same issue as above.... = solved as above 	

    // calling phase-4 Advance Particles kernel...
    AdvanceParticles_kernel <<< numBlocks, threads_per_block >>> (d_cells, d_cnumPars, numCells, timeStep);	  	
	//AdvanceParticles_kernel <<< numBlocks, threads_per_block >>> (cells, cnumPars, numCells, timeStep);	  	
							   // same issue as above.... = solved as above
}

int main (int argc, char **argv)
{

	// argv [1] is the input file name.
    
	int framenum = atoi(argv[1]);
	char *inputFile = argv [2];
	char *outputFile = argv [3];
	unsigned int hTimer;	
	double kernel_exec_time;
	double total_exec_Sum = 0;

	
	
	//InitSim(inputFile);

	
       // kernel configuration...
       const int threads_per_block = 256;
       const int numBlocks = numCells / threads_per_block + (numCells % threads_per_block == 0 ? 0:1);    
       printf ("Num Blocks : %d and Num Threads Per Block : %d \n", numBlocks, threads_per_block);       	   
	
    
	// Launch Kernels...
      // const int framenum = 100;
	
	
//	memset(cnumPars, 0, numCells*sizeof(int));

	//if (i ==0)
			InitSim(inputFile);
	

	CopyMemHtoD ();
	for ( int  i = 0; i < framenum; i++)
	{
	// For each frame iteration do the following...
				
		//const char* in = inputFile.c_str();	
		
		

		// copy updated memory to GPU
		
		cutilCheckError( cutCreateTimer(&hTimer) );
		cutCreateTimer(&hTimer);
		cutilCheckError( cutStartTimer(hTimer) );	

		/////////////////// Lauch Computation Kernels /////////////////
		AdvanceFrame_GPU (numBlocks, threads_per_block, i);
		///////////////////////////////////////////////////////////////
		
				kernel_exec_time = cutGetTimerValue(hTimer);
		
		
		// copy back memory	
		//CopyMemDtoH ();	
		cutilSafeCall (cudaThreadSynchronize ());
		
		cutilSafeCall( cudaMemcpy(cells, d_cells, numCells*sizeof(Cell), cudaMemcpyDeviceToHost) );
		cutilSafeCall( cudaMemcpy(cnumPars, d_cnumPars, numCells*sizeof(int), cudaMemcpyDeviceToHost) ); 
		
		//std::cout<<"cell [10].x "<<cells[2].p[10].x<<" : d_cell [10].x "<<d_cells[2].p[10].x<<std::endl; 
		
		std::string fName(outputFile);				
		char numstr[21]; // enough to hold all numbers up to 64-bits
		//result = name + itoa(age, numstr, 10);
				
		fName += itoa (i, numstr, 10);
		//strcat(outputFile, buffer);
	
		const char* out = fName.c_str();	
		std::cout<<"output File Name is =" <<out<<std::endl;
		//cutilSafeCall( cudaMemcpy(cells, d_cells, numCells*sizeof(Cell), cudaMemcpyDeviceToHost) );
		SaveFile(out);
		
		//d_cells = cells;
		//cnumPars = d_cnumPars;
		
		//cutilSafeCall( cudaMemcpy(d_cells,  cells,   numCells*sizeof(Cell), cudaMemcpyHostToDevice) );   
		//cutilSafeCall( cudaMemcpy(d_cells,  cells,   numCells*sizeof(Cell), cudaMemcpyHostToDevice) );   
		//cutilSafeCall( cudaMemcpy(d_cnumPars,  cnumPars,   numCells*sizeof(int), cudaMemcpyHostToDevice) );
		
			total_exec_Sum += kernel_exec_time;
    	}
		
		cutilSafeCall( cudaMemcpy(cells, d_cells, numCells*sizeof(Cell), cudaMemcpyDeviceToHost) );     
		cutilSafeCall( cudaMemcpy(cells2, d_cells2, numCells*sizeof(Cell), cudaMemcpyDeviceToHost) );     
		cutilSafeCall( cudaMemcpy(cnumPars, d_cnumPars, numCells*sizeof(int), cudaMemcpyDeviceToHost) ); 
		cutilSafeCall( cudaMemcpy(cnumPars2, d_cnumPars2, numCells*sizeof(int), cudaMemcpyDeviceToHost) );   


						
		
	
	// Save Result back
		//SaveFile (outputFile);

	// Validate Result...
	
	//int errCount = ValidateResult (outputFile, parsecFile, sqrSum);		

	//printf ("Error Count : %d and sqrSum : %f\n", errCount, sqrSum);
	printf("...fluidanimate Kernel(s) Execution Time (Computation Time)    : %f msec\n", total_exec_Sum);
	
	MemFree ();
}	



/*int main(int argc, char **argv){

     Cell *cells;
     Cell *d_cells;

     if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
	cutilDeviceInit(argc, argv);
     else
	cudaSetDevice( cutGetMaxGflopsDeviceId() );

     printf ("NUM CELLS %d \n", NUM_CELLS); 
     printf ("CELL Size %d \n", (int)sizeof (Cell));		 	 
     printf ("Alloc Size %d \n", (NUM_CELLS*sizeof(Cell)));
     
     cells = (Cell*) malloc(NUM_CELLS*sizeof(Cell));

     printf ("CPU Cell array size %d \n", (int)sizeof (cells));	

     cutilSafeCall( cudaMalloc((void **)&d_cells,  (NUM_CELLS*sizeof(Cell)) ) );

     printf ("GPU Cell array size %d \n", (int)sizeof (d_cells));	

     initData (cells);

     cutilSafeCall( cudaMemcpy(d_cells,  cells,   NUM_CELLS*sizeof(Cell), cudaMemcpyHostToDevice) );   
	
     int threads_per_block = 512;
     int numBlocks = NUM_CELLS / threads_per_block + (NUM_CELLS % threads_per_block == 0 ? 0:1);    

     printf ("Num Blocks : %d and Num Threads Per Block : %d \n", numBlocks, threads_per_block);	 
     // calling kernel
     cppTest_kernel<<< numBlocks, threads_per_block >>> (d_cells, NUM_CELLS);	

     cutilSafeCall( cudaThreadSynchronize() ); 
     printf ("After Kernel");		

     cutilSafeCall( cudaMemcpy(cells, d_cells, NUM_CELLS*sizeof(Cell), cudaMemcpyDeviceToHost) );     


     ValidateOutput (cells);

     free (cells);
     cutilSafeCall( cudaFree(d_cells)  );
 
  return 0;    
}
*/	
          		
	
	


	
