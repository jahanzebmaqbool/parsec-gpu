#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define fptype float

 typedef struct OptionData_ {
    fptype s;          // spot price
    fptype strike;     // strike price
    fptype r;          // risk-free interest rate
    fptype divq;       // dividend rate
    fptype v;          // volatility
    fptype t;          // time to maturity or option expiration in years 
	                   //     (1yr = 1.0, 6mos = 0.5, 3mos = 0.25, ..., etc)  
    char OptionType;   // Option type.  "P"=PUT, "C"=CALL
    fptype divs;       // dividend vals (not used in this test)
    fptype DGrefval;   // DerivaGem Reference Value
  } OptionData;

  OptionData *data;
  fptype *prices;
  int numOptions;

   int main (int argc, char *argv[])
   {

     FILE *file;
     FILE *out;
     int i;
     int loopnum;
     fptype * buffer;
     int * buffer2;
     int rv;
     int rv2;

//     nThreads = atoi(argv[1]);
     char *inputFile = argv[1];
     char *outputFile = argv[2];
     

    //Read input data from file
    file = fopen(inputFile, "r");
    out = fopen(outputFile, "w");

    if(file == NULL) {
      printf("ERROR: Unable to open file `%s'.\n", inputFile);
      exit(1);
    }
    rv = fscanf(file, "%i", &numOptions);
    if(rv != 1) {
      printf("ERROR: Unable to read from file `%s'.\n", inputFile);
      fclose(file);
      exit(1);
    }
    
    // alloc spaces for the option data
    data = (OptionData*)malloc(numOptions*sizeof(OptionData));
    prices = (fptype*)malloc(numOptions*sizeof(fptype));
    for ( loopnum = 0; loopnum < numOptions; ++ loopnum )
    {
        rv = fscanf(file, "%f %f %f %f %f %f %c %f %f", &data[loopnum].s, &data[loopnum].strike, &data[loopnum].r, &data[loopnum].divq, &data[loopnum].v, &data[loopnum].t, &data[loopnum].OptionType, &data[loopnum].divs, &data[loopnum].DGrefval);

        if(rv != 9) {
          printf("ERROR: Unable to read from file `%s'.\n", inputFile);
          fclose(file);
          exit(1);
        }

	
      rv2 = fprintf(out, "%.18f\n", data[loopnum].strike);
      if(rv2 < 0) {
        printf("ERROR: Unable to write to file `%s'.\n", outputFile);
        fclose(file);
        exit(1);
      }

    }
    rv = fclose(file);
    rv2 = fclose(out);
    if(rv != 0) {
      printf("ERROR: Unable to close file `%s'.\n", inputFile);
      exit(1);
    }
    
    if(rv2 != 0) {
      printf("ERROR: Unable to close file `%s'.\n", outputFile);
      exit(1);
    }	

   return 0;
}


