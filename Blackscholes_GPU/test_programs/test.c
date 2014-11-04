#include <stdio.h>

int main (int argc, char *argv[])
{
    FILE *in;
    FILE *out;
  
   printf ("%s", argv[1]);
//    in = fopen(argv[1], "r"); 
    out = fopen(argv[1], "w+");
	
  //  fprintf (out, "%s", argv[1]);

  return 0;
}
