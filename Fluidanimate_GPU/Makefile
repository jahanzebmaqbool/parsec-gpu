################################################################################
#
# Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to NVIDIA ownership rights under U.S. and 
# international Copyright laws.  
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
################################################################################
#
# Build script for project
#
################################################################################

# Exporting Required VARS

#export ROOTDIR=/home/jahanzeb/NVIDIA_GPU_Computing_SDK/C/src
#export SRCDIR=`pwd`
#export BINDIR=`pwd`
export verbose=1

#export ROOTOBJDIR =`pwd`/obj
#LIBDIR     := $(ROOTDIR)/../lib
#COMMONDIR  := $(ROOTDIR)/../common
#SHAREDDIR  := $(ROOTDIR)/../../shared/

# Compilers
#NVCC       := $(CUDA_INSTALL_PATH)/bin/nvcc 
#CXX        := g++
#CC         := gcc
#LINK       := g++ -fPIC

# Includes
#INCLUDES  += -I. -I$(CUDA_INSTALL_PATH)/include -I$(COMMONDIR)/inc -I$(SHAREDDIR)/inc
#INLCUDES += -I$(ROOTDIR)/../common/inc -I/usr/local/cuda/include
#LIBS = -L$(ROOTDIR)/../lib -lcutil -lshrutil_x86_64 -lglut
#LIBS += -L$(CUDA_LIB_PATH) -lcudart -lcuda

################################################################################

# Add source files here
EXECUTABLE	:= cppTest
# CUDA source files (compiled with cudacc)
CUFILES		:= cppTest.cu
# CUDA dependency files
#CU_DEPS		:= cppTest_kernel.cuh


CU_DEPS		:= \
	RebuildGrid_kernel.cuh \
	ComputeForces_kernel.cuh \
	ProcessCollisions_kernel.cuh \
	AdvanceParticles_kernel.cuh

#CU_DEPS         := RebuildGrid_kernel.cuh
#CU_DEPS          := ComputeForces_Kernel.cuh
#CU_DEPS         := $(CU_DEPS) ProcessCollision_kernel.cuh
#CU_DEPS         := $(CU_DEPS) AdvanceParticles_kernel.cuh

#CU_DEPS         += ComputeForces_kernel.cuh
#CU_DEPS         += ProcessCollisions_kernel.cuh
#CU_DEPS         += AdvanceParticles_kernel.cuh
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= 
# Need good occupancy
CUDACCFLAGS     := -po maxrregcount=16

################################################################################

#all:
#	nvcc --ptxas-options=-v $(INLCUDES) $(LIBS) $(CUFILES) $(CCFILES) $(CUDACCFLAGS)

# Rules and targets

#include $(ROOTDIR)/../common/common.mk
include common.mk

