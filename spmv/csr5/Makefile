#compilers
#GLOBAL_PARAMETERS
VALUE_TYPE = double
NUM_RUN = 1000

#ENVIRONMENT_PARAMETERS
#CUDA_PARAMETERS
#NVCC_FLAGS = -O3  -w -m64 -arch=sm_30
NVCC_FLAGS = -O3 -w -m64 -gencode=arch=compute_61,code=sm_61\  #-ccbin /home/tujinxing/gcc-7.5.0/bin
# CUDA_INSTALL_PATH = /usr/local/cuda
# CUDA_INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I/usr/local/cuda/samples/common/inc
CUDA_INSTALL_PATH = /home/tujinxing/cuda_10.1
CUDA_INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I/home/tujinxing/cuda_10.1/samples/common/inc
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart

cuda:
	nvcc $(NVCC_FLAGS) main.cu -o spmv $(CUDA_INCLUDES) $(CUDA_LIBS) -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)