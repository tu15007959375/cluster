################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/Options.cu \
../src/SpMV.cu \
../src/SpMVCSR.cu \
../src/main.cu 

CU_DEPS += \
./src/Options.d \
./src/SpMV.d \
./src/SpMVCSR.d \
./src/main.d 

OBJS += \
./src/Options.o \
./src/SpMV.o \
./src/SpMVCSR.o \
./src/main.o 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -I"/home/yongchao/cuda-workspace/LightSpMV-1.0" -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -I"/home/yongchao/cuda-workspace/LightSpMV-1.0" -G -g -O0 --compile --relocatable-device-code=true -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


