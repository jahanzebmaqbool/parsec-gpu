/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */

/*
 * This sample evaluates fair call and put prices for a
 * given set of European options by Black-Scholes formula.
 * See supplied whitepaper for more explanations.

 * Original Author : NVIDIA Corporation & PARSEC 

 * Modified By : Jahanzeb Maqbool | NUST School of Electrical Engineering and Computer Science
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <string.h>
#include <cutil_inline.h>

#include "OptionDataStruct.h"

  int numOptions;
  //OptionData* h_data;
        float* h_s;          // spot price
        float* h_strike;     // strike price
        float* h_r;          // risk-free interest rate
        float* h_divq;       // dividend rate
        float* h_v;          // volatility
        float* h_t;          // time to maturity or option expiration in years 
                          //     (1yr = 1.0, 6mos = 0.5, 3mos = 0.25, ..., etc)  
        char* h_OptionType;  // Option type.  "P"=PUT, "C"=CALL
        float* h_divs;       // dividend vals (not used in this test)
        float* h_DGrefval;   // DerivaGem Reference Value


////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////

extern "C" void BlackScholesCPU(float* h_s, float* h_strike, float* h_r, float* h_v, float* h_t, char* h_OptionType,
								//OptionData* data,
							    float *h_prices_CPU,
								int numOptions );

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////

#include "BlackScholes_kernel.cuh"

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////

#define NUM_ITERATIONS 100
  
//////////////////////////////////////////////////////////////////////////////


//__global__ void BlackScholesGPU (OptionData *data, float *OptionPrices, int numOptions); // array of OptionData structs.
__global__ void BlackScholesGPU (float d_s, float d_strike, float d_r, float d_v, float d_t, char d_OptionType, float *OptionPrices, int numOptions);

void ReadInputFile (char* inputFile)
{

     //printf("...Struct size %d.\n", sizeof(OptionData));
     FILE* file;	
     int loopnum ;
     int rv ;
     
     //char inputDir[] = "./input/";
     //char inputFile[] = *input;     
		
     //strcat (inputDir, inputFile);
     //strcpy (inputFile, inputDir);	
   

     printf("...Reading input data in CPU mem %s.\n", inputFile);
     //Read input data from file
     file = fopen(inputFile, "r");
     

     if(file == NULL) {
       printf("ERROR: Unable to open file `%s'.\n", inputFile);
       exit(1);
     }
     rv = fscanf(file, "%i", &numOptions);
     
     
	 h_s = (float*) malloc(numOptions*sizeof(float));
	 h_strike = (float*) malloc(numOptions*sizeof(float));
	 h_r = (float*) malloc(numOptions*sizeof(float));
	 h_divq = (float*) malloc(numOptions*sizeof(float));
	 h_v = (float*) malloc(numOptions*sizeof(float));
	 h_t = (float*) malloc(numOptions*sizeof(float));
	 h_OptionType = (char*) malloc(numOptions*sizeof(char));
	 h_divs = (float*) malloc(numOptions*sizeof(float));
	 h_DGrefval = (float*) malloc(numOptions*sizeof(float));
	
     	
     if(rv != 1) {
       printf("ERROR: Unable to read from file `%s'.\n", inputFile);
       fclose(file);
       exit(1);
     }
    
       // READING INPUT FROM FILE, and store it in array of STURCT.
	

	for ( loopnum = 0; loopnum < numOptions; ++ loopnum )
        {
    	    rv = fscanf(file, "%f %f %f %f %f %f %c %f %f", &h_s[loopnum], &h_strike[loopnum], &h_r[loopnum], &h_divq[loopnum], &h_v[loopnum], &h_t[loopnum],
			&h_OptionType[loopnum], &h_divs[loopnum], &h_DGrefval[loopnum]);
            //if(rv != 9) {
            //   printf("ERROR: Unable to read from file `%s'.\n", inputFile);
             //  fclose(file);
            //   exit(1);
            //}
         }
        //rv = fclose(file);
       

	printf("SUCCESS: Read data from input file `%s'.\n", inputFile);	
	printf("Going to close file... `%s'.\n", inputFile);
	printf("... Func End ....%d\n", numOptions);
	//file = NULL;	
	
	//free (file);
	rv = fclose(file);
	 if(rv != 0) {
 	     printf("ERROR: Unable to close file `%s'.\n", inputFile);
 	     exit(1);
        }
	
        //if(rv != 0) {
         // printf("ERROR: Unable to close file `%s'.\n", inputFile);
          //exit(1);
        //}
	
}

////////////////////////////////////////////////////////////////////////////////
// WRITE OUTPUT RESULT FILE 
////////////////////////////////////////////////////////////////////////////////

void writePriceResults (char* filename, float* prices)
{	
	FILE* file;
	int rv;
	int loopIter;
	
	file = fopen(filename, "w+");
	if(file == NULL) {
	      printf("ERROR: Unable to open file `%s'.\n", filename);
	      exit(1);
	}	
	
	for (loopIter = 0 ; loopIter < numOptions ; loopIter ++)	
		rv = fprintf(file, "%.18f\n", prices[loopIter]);
	
	if(rv < 0) {
           printf("ERROR: Unable to write to file `%s'.\n", filename);
           fclose(file);
           exit(1);
        }

	fclose (file);       	
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
     
     //FILE *out;
     // Host Data...
     //OptionData *h_data;
     float *h_prices; 	
     float *h_prices_CPU;
     // GPU device Data...
     //OptionData *d_data;
	   
		float *d_s;          // spot price
        float *d_strike;     // strike price
        float *d_r;          // risk-free interest rate
        float *d_v;          // volatility
        float *d_t;          // time to maturity or option expiration in years 
        char *d_OptionType;  // Option type.  "P"=PUT, "C"=CALL
        
	 
     float *d_prices;   	
 
     double
     delta, ref, sum_delta, sum_ref, max_delta, L1norm, gpuTime;
     unsigned int hTimer;
    
	struct timeval c_start, c_end;
    
	float cAlocTime, gAlocTime, MemTransferTime1, MemTransferTime2;
	
	long  c_mtime, c_seconds, c_useconds, cpuTime,  caloc_time_usec, caloc_time_second, galoc_time_usec
	,galoc_time_second, trans_time_usec1, trans_time_second1, trans_time_usec2, trans_time_second2;    

	struct timeval memTransferStart1, memTransferEnd1, memTransferStart2, memTransferEnd2, cpuMemAlocStart, cpuMemAlocEnd, gpuMemAlocStart, gpuMemAlocEnd;
	 
	 
	 //clock_t cpu_start, cpu_end;
     int i;

	 
	
     char *inputFile = argv[1];
     char *outputFile = argv[2];
 
	 
	 printf(">>>> %s >>>>> %s \n", inputFile, outputFile); 
	 	 
     if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	 else
		cudaSetDevice( cutGetMaxGflopsDeviceId() );
		
    cutilCheckError( cutCreateTimer(&hTimer) );
	cutCreateTimer(&hTimer);

        printf("...Initializing data...\n");
        
	ReadInputFile (inputFile);
	
	printf("... allocating CPU memory for options\n"); 
	gettimeofday (&cpuMemAlocStart, NULL);
	h_prices = (float*) malloc(numOptions*sizeof(float));
	h_prices_CPU = (float*) malloc(numOptions*sizeof (float));		
	gettimeofday (&cpuMemAlocEnd, NULL);
	
	cAlocTime = ( (cpuMemAlocEnd.tv_usec - cpuMemAlocStart.tv_usec));  // time in microsecond (10E-3)
	//caloc_time_usec = (cpuMemAlocEnd.tv_usec - cpuMemAlocStart.tv_usec);  
	//caloc_time_second = cpuMemAlocEnd.tv_sec - cpuMemAlocStart.tv_sec; 
	//cAlocTime = (caloc_time_second*1000) + (caloc_time_usec/1000) + 0.5;
	
	printf("...allocating GPU memory for options.\n");

			// dummy alloc....	
					cutilSafeCall( cudaMalloc((void **)&d_s,  (numOptions*sizeof(float)) ) );
					cutilSafeCall( cudaMalloc((void **)&d_strike,  (numOptions*sizeof(float)) ) );
					cutilSafeCall( cudaMalloc((void **)&d_r,  (numOptions*sizeof(float)) ) );					
					cutilSafeCall( cudaMalloc((void **)&d_v,  (numOptions*sizeof(float)) ) );
					cutilSafeCall( cudaMalloc((void **)&d_t,  (numOptions*sizeof(float)) ) );
					cutilSafeCall( cudaMalloc((void **)&d_OptionType,  (numOptions*sizeof(char)) ) );
									
					cutilSafeCall( cudaMalloc((void **)&d_prices,  (numOptions*sizeof(float)) ) );
		
		printf("...dummy alloc done\n");

		
					cutilSafeCall( cudaFree(d_s)  );
					cutilSafeCall( cudaFree(d_strike)  );
					cutilSafeCall( cudaFree(d_v)  );
					cutilSafeCall( cudaFree(d_t)  );
					cutilSafeCall( cudaFree(d_OptionType)  );
					cutilSafeCall( cudaFree(d_prices) );
			//////////////////////////////////////
			
			
		
	

	gettimeofday (&gpuMemAlocStart, NULL);
		cutilSafeCall( cudaMalloc((void **)&d_s,  (numOptions*sizeof(float)) ) );
		cutilSafeCall( cudaMalloc((void **)&d_strike,  (numOptions*sizeof(float)) ) );
		cutilSafeCall( cudaMalloc((void **)&d_r,  (numOptions*sizeof(float)) ) );
		cutilSafeCall( cudaMalloc((void **)&d_v,  (numOptions*sizeof(float)) ) );
		cutilSafeCall( cudaMalloc((void **)&d_t,  (numOptions*sizeof(float)) ) );
		cutilSafeCall( cudaMalloc((void **)&d_OptionType,  (numOptions*sizeof(char)) ) );

					
		cutilSafeCall( cudaMalloc((void **)&d_prices,  (numOptions*sizeof(float)) ) );				
	gettimeofday (&gpuMemAlocEnd, NULL);	
	
	gAlocTime = ((gpuMemAlocEnd.tv_usec - gpuMemAlocStart.tv_usec)); // time in microsecond (10E-3)	
	//galoc_time_usec = (gpuMemAlocEnd.tv_usec - gpuMemAlocStart.tv_usec);  
	//galoc_time_second = gpuMemAlocEnd.tv_sec - gpuMemAlocStart.tv_sec; 
	//gAlocTime = (galoc_time_second*1000) + (galoc_time_usec/1000) + 0.5;

	printf("...Data init done.\n\n");
	printf("...copying input data to GPU mem.\n");
  
	gettimeofday (&memTransferStart1, NULL);
        cutilSafeCall( cudaMemcpy(d_s,  h_s,   numOptions*sizeof(float), cudaMemcpyHostToDevice) );
		cutilSafeCall( cudaMemcpy(d_strike,  h_strike,   numOptions*sizeof(float), cudaMemcpyHostToDevice) );
		cutilSafeCall( cudaMemcpy(d_r,  h_r,   numOptions*sizeof(float), cudaMemcpyHostToDevice) );
		cutilSafeCall( cudaMemcpy(d_v,  h_v,   numOptions*sizeof(float), cudaMemcpyHostToDevice) );
		cutilSafeCall( cudaMemcpy(d_t,  h_t,   numOptions*sizeof(float), cudaMemcpyHostToDevice) );
		cutilSafeCall( cudaMemcpy(d_OptionType,  h_OptionType,   numOptions*sizeof(char), cudaMemcpyHostToDevice) );
		
        cutilSafeCall( cudaMemcpy(d_prices,  h_prices,   numOptions*sizeof(float), cudaMemcpyHostToDevice) );		
	gettimeofday (&memTransferEnd1, NULL);	
	
    //MemTransferTime1 = ( (memTransferEnd1.tv_usec - memTransferStart1.tv_usec)); // time in microsecond (10E-6)
	trans_time_usec1 = (memTransferEnd1.tv_usec - memTransferStart1.tv_usec);  
	trans_time_second1 = memTransferEnd1.tv_sec - memTransferStart1.tv_sec; 
	MemTransferTime1 = (trans_time_second1*1000) + (trans_time_usec1/1000);


        printf("...Executing Black-Scholes GPU kernel (%i iterations)...\n", NUM_ITERATIONS);
        cutilSafeCall( cudaThreadSynchronize() );
		cutilCheckError( cutResetTimer(hTimer) );
        cutilCheckError( cutStartTimer(hTimer) );
        

	// calling Kernel for number of iteration times....
        for(int i = 0; i < NUM_ITERATIONS; i++){

            BlackScholesGPU<<<480, 128>>>( d_s, d_strike, d_r, d_v, d_t, d_OptionType, d_prices, numOptions );

            cutilCheckMsg("...BlackScholesGPU() execution failed\n");
        }
	
	cutilSafeCall( cudaThreadSynchronize() );
        //			cudaThreadSynchronize();
    cutilCheckError( cutStopTimer(hTimer) );
    gpuTime = cutGetTimerValue(hTimer) / NUM_ITERATIONS;

	
	// copy back results to cput....
	gettimeofday (&memTransferStart2, NULL);
       cutilSafeCall( cudaMemcpy(h_prices, d_prices, numOptions*sizeof(float), cudaMemcpyDeviceToHost) );
	gettimeofday (&memTransferEnd2, NULL);	
	
//    MemTransferTime2 = ( (memTransferEnd2.tv_usec - memTransferStart2.tv_usec)); // time in microsecond (10E-6)
	trans_time_usec2 = (double)(memTransferEnd2.tv_usec - memTransferStart2.tv_usec);  
	trans_time_second2 = memTransferEnd2.tv_sec - memTransferStart2.tv_sec; 
	MemTransferTime2 = (trans_time_second2*1000) + (trans_time_usec2/1000);

			
	printf("...Options count             : %i     \n", numOptions);
	//printf ("...BlackScholesGPU() Host Mem Aloc Time    : %f msec\n", (double)((cpuMemAlocEnd - cpuMemAlocStart)/CLOCKS_PER_SEC)/1000 );
	printf("...BlackScholesGPU() Host Mem Aloc Time    : %f msec\n", cAlocTime/1000);// time in millisecond (10E-3)
	printf("...BlackScholesGPU() DEVICE Mem Aloc Time    : %f msec\n", gAlocTime/1000);
	printf("...BlackScholesGPU() Mem Transfer H_to_D Time    : %f msec\n", MemTransferTime1);
    printf("...BlackScholesGPU() Kernel Execution Time (Computation Time)    : %f msec\n", gpuTime);
	printf("...BlackScholesGPU() Mem Transfer D_to_H Time    : %f msec\n", MemTransferTime2);
	
	printf("#.#.#.# Total GPU Time   : %f msec\n", (gpuTime + MemTransferTime1 + MemTransferTime2 + gAlocTime/1000));
	
	
        printf("...Effective memory bandwidth: %f GB/s\n", ((double)(5 * numOptions * sizeof(float)) * 1E-9) / (gpuTime * 1E-3));
        printf("...Gigaoptions per second    : %f     \n\n", ((double)(2 * numOptions) * 1E-9) / (gpuTime * 1E-3));
    
        printf("...Reading back GPU results...\n");
        //Read back GPU results to compare them to CPU results
        //cutilSafeCall( cudaMemcpy(h_prices, d_prices, numOptions*sizeof(float), cudaMemcpyDeviceToHost) );
	
       	
			
		printf("...Checking the results...\n\n");
        printf("...running CPU calculations.\n");
        


	//cpu_start = clock();	
	//gettimeofday (&cpuTimeStart, NULL);
	//Calculate options values on CPU


	gettimeofday(&c_start, NULL);
	
	for(int i = 0; i < NUM_ITERATIONS; i++){
		BlackScholesCPU( h_s, h_strike, h_r, h_v, h_t, h_OptionType,  h_prices_CPU , numOptions);
	}

    gettimeofday(&c_end, NULL);

	c_seconds = (double)(c_end.tv_usec - c_start.tv_usec);  
	c_useconds = c_end.tv_sec - c_start.tv_sec; 
	c_mtime = (c_seconds*1000) + (c_useconds/1000) + 0.5;
	
//    c_seconds  = c_end.tv_sec  - c_start.tv_sec;
//    c_useconds = c_end.tv_usec - c_start.tv_usec;

//   c_mtime = ((c_seconds) * 1000 + (c_useconds/1000.0)) + 0.5;

    //printf("CPU Elapsed time: %ld milliseconds\n", c_mtime);

	
	//cpu_end = clock();
	//gettimeofday (&cpuTimeEnd, NULL);	


	//printf ("CPU Time required for execution of %d frames is : %f seconds \n", NUM_ITERATIONS,  (double)((cpu_end-cpu_start)/CLOCKS_PER_SEC));

	//cpuTime = cpuTimeEnd.tv_usec - cpuTimeStart.tv_usec; // time in microsecond (10E-6)
	//cpuTime = ((double) ((cpuTimeEnd.tv_usec*1000) - (cpuTimeStart.tv_usec*1000))); // time in milisecond (10E-3)	
	
	printf("...CPU TIME TAKEN BY %d ITERATIONs ... %ld msec\n", NUM_ITERATIONS , c_mtime); // time in millisecond (10E-3)	
	

	printf("...Comparing the results...\n");
        //Calculate max absolute difference and L1 distance
        //between CPU and GPU results
        sum_delta = 0;
        sum_ref   = 0;
        max_delta = 0;
        for(i = 0; i < numOptions; i++){
            ref   = h_prices_CPU[i];
            delta = fabs(h_prices_CPU[i] - h_prices[i]);
            if(delta > max_delta)
		 max_delta = delta;
            sum_delta += delta;
            sum_ref   += fabs(ref);
        }
        L1norm = sum_delta / sum_ref;
        printf("...L1 norm: %E\n", L1norm);
        printf("...Max absolute error: %E\n", max_delta);
        printf((L1norm < 1e-6) ? "TEST PASSED\n" : "TEST FAILED\n");


    	printf("...Writing back Option Price Results to file %s...\n", outputFile);	
	// calling function to write back the results to the disk
	writePriceResults (outputFile, h_prices);	
	printf("...SUCCESS: Prices has been succesfully written into file `%s'.\n", outputFile);	
	

    	printf("...Shutting down...\n");
        printf("...releasing GPU memory.\n");
       
		cutilSafeCall( cudaFree(d_s)  );
		cutilSafeCall( cudaFree(d_strike)  );
		cutilSafeCall( cudaFree(d_v)  );
		cutilSafeCall( cudaFree(d_t)  );
		cutilSafeCall( cudaFree(d_OptionType)  );
		cutilSafeCall( cudaFree(d_prices) );

        printf("...releasing CPU memory.\n");
       
	   
		   
		free(h_s);
		free(h_strike);
		free(h_r);
		free(h_divq);
		free(h_v);
		free(h_t);
		free(h_OptionType);
		free(h_divs);
		free(h_DGrefval);
		
        free(h_prices);
        free(h_prices_CPU);
        cutilCheckError( cutDeleteTimer(hTimer) );
    	printf("...Shutdown done.\n");

    cudaThreadExit();
}	


