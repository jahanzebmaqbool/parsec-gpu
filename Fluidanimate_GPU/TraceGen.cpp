//Code originally written by Richard O. Lee
//Modified by Christian Bienia and Christian Fensch

#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <vector>
/*
#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif
*/

#define DEBUG false

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

// note: icc-optimized version of this class gave 15% more
// performance than our hand-optimized SSE3 implementation
class Vec3
{
public:
    float x, y, z;

    Vec3() {}
    Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

    float   GetLengthSq() const         { return x*x + y*y + z*z; }
    float   GetLength() const           { return sqrtf(GetLengthSq()); }
    Vec3 &  Normalize()                 { return *this /= GetLength(); }

    Vec3 &  operator += (Vec3 const &v) { x += v.x;  y += v.y; z += v.z; return *this; }
    Vec3 &  operator -= (Vec3 const &v) { x -= v.x;  y -= v.y; z -= v.z; return *this; }
    Vec3 &  operator *= (float s)       { x *= s;  y *= s; z *= s; return *this; }
    Vec3 &  operator /= (float s)       { x /= s;  y /= s; z /= s; return *this; }

    Vec3    operator + (Vec3 const &v) const    { return Vec3(x+v.x, y+v.y, z+v.z); }
    Vec3    operator - () const                 { return Vec3(-x, -y, -z); }
    Vec3    operator - (Vec3 const &v) const    { return Vec3(x-v.x, y-v.y, z-v.z); }
    Vec3    operator * (float s) const          { return Vec3(x*s, y*s, z*s); }
    Vec3    operator / (float s) const          { return Vec3(x/s, y/s, z/s); }
	
    float   operator * (Vec3 const &v) const    { return x*v.x + y*v.y + z*v.z; }
};

////////////////////////////////////////////////////////////////////////////////

// there is a current limitation of 16 particles per cell
// (this structure use to be a simple linked-list of particles but, due to
// improved cache locality, we get a huge performance increase by copying
// particles instead of referencing them)
struct Cell
{
	Vec3 p[16];
	Vec3 hv[16];
	Vec3 v[16];
	Vec3 a[16];
	float density[16];
};

struct ParticlePosition 
{
	float x, y, z;
};

////////////////////////////////////////////////////////////////////////////////

int globalParticleSize = 0;

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


std::vector <ParticlePosition> particles;

//ParticlePosition* particles = 0;


std::string inputFile = "";
std::string outputFile = "";











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

void ReadFile(char const *fileName)
{
	if(DEBUG)
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
	viscosityCoeff = viscosity * coeff3 * particleMass;

	Vec3 range = domainMax - domainMin;
	nx = (int)(range.x / h);
	ny = (int)(range.y / h);
	nz = (int)(range.z / h);
	assert(nx >= 1 && ny >= 1 && nz >= 1);
	numCells = nx*ny*nz;
	if(DEBUG)
	std::cout << "Number of cells: " << numCells << std::endl;
	delta.x = range.x / nx;
	delta.y = range.y / ny;
	delta.z = range.z / nz;
	assert(delta.x >= h && delta.y >= h && delta.z >= h);

	//particles = new std::vector <ParticlePosition> ();
	//particles = new ParticlePosition [origNumParticles];
	
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

		//if (px == 0 && py == 0 && pz ==0)
		//	continue;
			

		ParticlePosition pos;
		pos.x = px ;//* 1000 ;
		pos.y = py ;//* 1000 ;
		pos.z = pz ;//* 1000 ;
		
		particles.push_back (pos);
			
	    //particles [i].x = px;
		//particles [i].y = py;
		//particles [i].z = pz;
		
		//std::cout <<"x = "<<particles [i].x<<" , y = "<<particles [i].y<<" , z = "<<particles [i].z<<std::endl;
		if(DEBUG)
		std::cout <<"x = "<<pos.x<<" , y = "<<pos.y<<" , z = "<<pos.z<<std::endl;
	}
	if(DEBUG)
	std::cout<<"All data copie to Particles Array...." <<std::endl;
	//std::cout << "Number of particles: " << numParticles << " (" << origNumParticles-numParticles << " skipped)" << std::endl;
}

////////////////////////////////////////////////////////////////////////////////

void SaveFile(char const *fileName)
{
	if(DEBUG)
	std::cout << "Saving file \"" << fileName << "\"..." << std::endl;
	int length = particles.size();
	if(DEBUG)
	std::cout << "Size of Particle Array = " << length <<std::endl;
	

	//ofstream fout;
	//fout.open("output.txt");
	std::ofstream fout (fileName, std::ios::app);
	  	
	for (int i = 0; i < length; i++) {		
		fout << particles [i].x <<" "<< particles [i].y <<" "<< particles [i].z<< "\n";	
	}
	
	
	
	
	//int num = 150; 
	//char name[] = "John Doe";
	//fout << "Here is a number: " << num << "\n";
	//fout << "Now here is a string: " << name << "\n";
	
	
	fout << "#" << "\n";
		
	fout << std::flush;
	fout.close();	
	
	globalParticleSize += length;
	particles.clear ();

}

/////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
/*
int WriteIFrITBasicParticlesBinFile(
    int n, // Number of particles 
    float xl, float yl, float zl, float xh, float yh, float zh, // Bounding box 
    float *x, float *y, float *z, // Particle positions (can be double) 
    //float *attr1, float *attr2, float *attr3, // Particle attributes 
    char *filename) // Name of the file 
{
   int i, ntemp; FILE *F;
   F = fopen(filename,"w"); if(F == NULL) return 1;
   ntemp = 4;
   fwrite(&ntemp,4,1,F);
   fwrite(&n,4,1,F);
   fwrite(&ntemp,4,1,F);
   ntemp = 24;
   fwrite(&ntemp,4,1,F);
   fwrite(&xl,4,1,F);
   fwrite(&yl,4,1,F);
   fwrite(&zl,4,1,F);
   fwrite(&xh,4,1,F);
   fwrite(&yh,4,1,F);
   fwrite(&zh,4,1,F);
   fwrite(&ntemp,4,1,F);
   ntemp = sizeof(x[0])*n;
   fwrite(&ntemp,4,1,F); fwrite(x,sizeof(x[0]),n,F); fwrite(&ntemp,4,1,F);
   fwrite(&ntemp,4,1,F); fwrite(y,sizeof(y[0]),n,F); fwrite(&ntemp,4,1,F);
   fwrite(&ntemp,4,1,F); fwrite(z,sizeof(z[0]),n,F); fwrite(&ntemp,4,1,F);
   //ntemp = 4*n;
   //fwrite(&ntemp,4,1,F); fwrite(attr1,4,n,F); fwrite(&ntemp,4,1,F);
   //fwrite(&ntemp,4,1,F); fwrite(attr2,4,n,F); fwrite(&ntemp,4,1,F);
   //fwrite(&ntemp,4,1,F); fwrite(attr3,4,n,F); fwrite(&ntemp,4,1,F);
   //fclose(F);
   return 0;
}
*/







////////////////////////////////////////////////////////////////////////////////
	
	bool check = false;
	std::string suff = "";
void SaveFile2(const char* fileName, int iter) {
	
	int pSize = particles.size ();
	globalParticleSize += pSize;
	
	
	std::stringstream ss;
	std::string tempStr;
	ss << iter;
	ss >> tempStr;
	int length = tempStr.length();
	for	(int i = 0; i < 4-length; i++) {
		tempStr = "0"+tempStr;
	}
	
	//convert fileName to string and append suffix...
	std::string fName(fileName);
	fName = fName +"_"+ tempStr +".txt";
	
	//convert back appended fName to const char* and open file..
	
	const char* outFile = fName.c_str();	
	FILE *F = fopen(outFile,"w");
	
	
	//if (!check) {
		fprintf(F,"%d\n", pSize);
		fprintf(F,"%g %g %g %g %g %g\n", -0.065, -0.08, -0.065, 0.065, 0.1, 0.065);
	//}
	
	for(int i=0; i < pSize; i++) {
		fprintf(F,"%g %g %g\n", particles[i].x, particles[i].y, particles[i].z);
	}

	//fprintf(F,"#");
	
	fclose(F);
	particles.clear ();
	//check = true;
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
	
	
	int framenum = atoi(argv[1]);
	inputFile = argv[2];
	outputFile = argv[3];
	
	for (int i = 0; i < framenum; i ++) {
		
		std::string tempFName = inputFile;
		
		char numstr[21];
		itoa (i, numstr, 10);
		//strcat (tempFName, numstr);
		tempFName += numstr;
	
		const char* in = tempFName.c_str();	
		
		if(DEBUG)
		std::cout<<"Reading File :: "<<tempFName<<std::endl;	
		
		ReadFile (in);
		
		const char* out = outputFile.c_str();	
		//SaveFile (out);
		
		SaveFile2 (out, i+1);
	}

	
	std::cout<<"Global Length :: "<<globalParticleSize<<std::endl;
  return 0;
}


