#source files
NVCC_src = main.cu  Options.cu  SpMV.cu  SpMVCSR.cu

src_dir = ./src
objs_dir = objs
objs += $(patsubst %.cu, $(objs_dir)/%.cu.o, $(NVCC_src))

#GPU architecture
GPU_ARCH = sm_35
GPU_ARCH_FLAGS = -arch $(GPU_ARCH)
ifeq ($(GPU_ARCH), sm_35)
  GPU_ARCH_FLAGS += -DHAVE_SM_35
endif

#do NOT USE texture memory for single-precision floating point
#MACROS += -DNO_FLOAT_TEXTURE_MEMORY

#do NOT use texture memory for double-precision floating point
#MACROS += -DNO_DOUBLE_TEXTURE_MEMORY

#compiling parameters
CXX = nvcc
NVCC = nvcc
NVCCFLAGS = --cudart static -O3 -I. -I $(src_dir) $(GPU_ARCH_FLAGS) -rdc true --ptxas-options=-v -Xcompiler -funroll-loops -Xcompiler -Wall $(MACROS) -ccbin /home/tujinxing/gcc-7.5.0/bin
# NVCCFLAGS = --cudart static -O3 -I. -I $(src_dir) $(GPU_ARCH_FLAGS) -rdc true -Xcompiler -funroll-loops -Xcompiler -Wall $(MACROS)

EXEC = lightspmv

all: dir $(objs)
	$(NVCC) $(NVCCFLAGS) -o $(EXEC) $(objs) -lm
	strip $(EXEC)

dir:
	mkdir -p $(objs_dir)

$(objs_dir)/%.cu.o: $(src_dir)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean: 
	rm -rf $(EXEC) $(objs_dir) 

#dependency
SpMV.cu : SpMVCSR.h
