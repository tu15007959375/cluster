# Compiler
NVCC = nvcc

# Compiler flags
CFLAGS = -Xcudafe --diag_suppress=2464 -Xcompiler -fopenmp -O3 -use_fast_math  \

# Target executable
TARGET = diaspmv 

# Source files
SRCS = main.cu

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)
