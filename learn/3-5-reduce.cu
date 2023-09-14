#include <cuda_runtime.h>
#include <stdio.h>
#include "util.h"
#include <thrust/extrema.h>
//cpu的规约操作
int cpurecursiveReduce(int *data, int const size)
{
    if(size == 1)
        return data[0];
    int stride = size/2;
    if(size%2)
    {
        for(int i=0;i<stride;i++)
            data[i] +=data[i+stride];
        data[0] += data[size-1];
    }
    else
    {
        for(int i=0;i<stride;i++)
            data[i] +=data[i+stride];
    }
    return cpurecursiveReduce(data,stride);
}

//初始版本的归约操作,使用偶数线程处理数据，而奇数线程完全空转
//存在问题：浪费的线程太多
__global__ void reduce(int *data,int *out,int size)
{
    int tid = threadIdx.x;
    if(tid>=size)
        return;
    int *nowdata = data+blockIdx.x*blockDim.x;
    //带dim的都是固定大小，blockdim就是线程块的大小，即一个block中有多少thread
    for(int i=1;i<blockDim.x;i *=2)
    {
        if(tid%(2*i)==0)
        {
            nowdata[tid] += nowdata[tid+i];
        }
        __syncthreads();
    }
    if(tid==0)       
    {
        out[blockIdx.x]=nowdata[0]; 
    }
}

//第二版的归约
//0 <- 0+1
//使用前面的线程处理，充分利用线程数据
__global__ void reduce2(int *data,int *out,int size)
{
    int tid = threadIdx.x;
    if(tid>=size)
        return;
    int *nowdata = data+blockIdx.x*blockDim.x*2;
    //带dim的都是固定大小，blockdim就是线程块的大小，即一个block中有多少thread
    for(int i=1;i<blockDim.x*2;i *=2)
    {
        int index = 2 * i *tid;
		if (index < blockDim.x*2)
		{
			nowdata[index] += nowdata[index + i];
		}
        __syncthreads();
    }
    if(tid==0)       
    {
        out[blockIdx.x]=nowdata[0]; 
    }
}

//第三版的归约
//0 <- 0+1024
__global__ void reduce3(int *data,int *out,int size)
{
    int tid = threadIdx.x;
    int *nowdata = data+blockIdx.x*blockDim.x*2;
    //带dim的都是固定大小，blockdim就是线程块的大小，即一个block中有多少thread
    int stride = blockDim.x;
    for(int i=stride;i>0;i /= 2)
    {
        if(tid < i)
        {
            nowdata[tid] += nowdata[tid+i];
        }
        __syncthreads();
    }
    if(tid==0)       
    {
        out[blockIdx.x]=nowdata[0]; 
    }
}

//第四版的归约
//循环展开unrool2
//0 <- 0+1024
__global__ void reduce4(int *data,int *out,int size)
{
    int tid = threadIdx.x;
    unsigned int idx = blockDim.x*blockIdx.x*2+threadIdx.x;
    //带dim的都是固定大小，blockdim就是线程块的大小，即一个block中有多少thread
    if(idx+blockDim.x<size)
	{
		data[idx]+=data[idx+blockDim.x];

	}
	__syncthreads();
    int *nowdata = data+blockIdx.x*blockDim.x*2;
    int stride = blockDim.x;
    for(int i=stride/2;i>0;i /= 2)
    {
        if(tid < i)
        {
            nowdata[tid] += nowdata[tid+i];
        }
        __syncthreads();
    }
    if(tid==0)       
    {
        out[blockIdx.x]=nowdata[0]; 
    }
}
//全局内存写入共享内存提升速度
__global__ void reduceSmem(int * g_idata,int * g_odata,unsigned int n)
{
	//set thread ID
    __shared__ int smem[1024];
	unsigned int tid = threadIdx.x;
	//unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
	//boundary check
	if (tid >= n) return;
	//convert global data pointer to the
	int *idata = g_idata + blockIdx.x*blockDim.x;

    smem[tid]=idata[tid];
	__syncthreads();
	//in-place reduction in global memory
	if(blockDim.x>=1024 && tid <512)
		smem[tid]+=smem[tid+512];
	__syncthreads();
	if(blockDim.x>=512 && tid <256)
		smem[tid]+=smem[tid+256];
	__syncthreads();
	if(blockDim.x>=256 && tid <128)
		smem[tid]+=smem[tid+128];
	__syncthreads();
	if(blockDim.x>=128 && tid <64)
		smem[tid]+=smem[tid+64];
	__syncthreads();
	//write result for this block to global mem
	if(tid<32)
	{
		volatile int *vsmem = smem;
		vsmem[tid]+=vsmem[tid+32];
		vsmem[tid]+=vsmem[tid+16];
		vsmem[tid]+=vsmem[tid+8];
		vsmem[tid]+=vsmem[tid+4];
		vsmem[tid]+=vsmem[tid+2];
		vsmem[tid]+=vsmem[tid+1];

	}

	if (tid == 0)
		g_odata[blockIdx.x] = smem[0];

}
//第五版的归约
//循环展开unrool8+最后64线程规约使用volatile变量
//0 <- 0+1024+2048+。。。。+1024*7
__global__ void reduce5(int *data,int *out,int size)
{
    int tid = threadIdx.x;
    unsigned int idx = blockDim.x*blockIdx.x*8+threadIdx.x;
    //带dim的都是固定大小，blockdim就是线程块的大小，即一个block中有多少thread
    if(idx+7*blockDim.x<size)
	{
		int a1=data[idx];
		int a2=data[idx+blockDim.x];
		int a3=data[idx+2*blockDim.x];
		int a4=data[idx+3*blockDim.x];
		int a5=data[idx+4*blockDim.x];
		int a6=data[idx+5*blockDim.x];
		int a7=data[idx+6*blockDim.x];
		int a8=data[idx+7*blockDim.x];
		data[idx]=a1+a2+a3+a4+a5+a6+a7+a8;
	}
	__syncthreads();
    int *nowdata = data+blockIdx.x*blockDim.x*8;

    if(blockDim.x>=1024 && tid <512)
		nowdata[tid]+=nowdata[tid+512];
	__syncthreads();
	if(blockDim.x>=512 && tid <256)
		nowdata[tid]+=nowdata[tid+256];
	__syncthreads();
	if(blockDim.x>=256 && tid <128)
		nowdata[tid]+=nowdata[tid+128];
	__syncthreads();
	if(blockDim.x>=128 && tid <64)
		nowdata[tid]+=nowdata[tid+64];
	__syncthreads();
    if(tid<32)
	{
		volatile int *vmem = nowdata;
		vmem[tid]+=vmem[tid+32];
		vmem[tid]+=vmem[tid+16];
		vmem[tid]+=vmem[tid+8];
		vmem[tid]+=vmem[tid+4];
		vmem[tid]+=vmem[tid+2];
		vmem[tid]+=vmem[tid+1];

	}
    if(tid==0)       
    {
        out[blockIdx.x]=nowdata[0]; 
    }
}

__global__ void reduceUnroll4Smem(int * g_idata,int * g_odata,unsigned int n)
{
	//set thread ID
    __shared__ int smem[1024];
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockDim.x*blockIdx.x*4+threadIdx.x;
	//boundary check
	if (tid >= n) return;
	//convert global data pointer to the
    int tempSum=0;
	if(idx+3 * blockDim.x<=n)
	{
		int a1=g_idata[idx];
		int a2=g_idata[idx+blockDim.x];
		int a3=g_idata[idx+2*blockDim.x];
		int a4=g_idata[idx+3*blockDim.x];
		tempSum=a1+a2+a3+a4;

	}
    smem[tid]=tempSum;
	__syncthreads();
	//in-place reduction in global memory
	if(blockDim.x>=1024 && tid <512)
		smem[tid]+=smem[tid+512];
	__syncthreads();
	if(blockDim.x>=512 && tid <256)
		smem[tid]+=smem[tid+256];
	__syncthreads();
	if(blockDim.x>=256 && tid <128)
		smem[tid]+=smem[tid+128];
	__syncthreads();
	if(blockDim.x>=128 && tid <64)
		smem[tid]+=smem[tid+64];
	__syncthreads();
	//write result for this block to global mem
	if(tid<32)
	{
		volatile int *vsmem = smem;
		vsmem[tid]+=vsmem[tid+32];
		vsmem[tid]+=vsmem[tid+16];
		vsmem[tid]+=vsmem[tid+8];
		vsmem[tid]+=vsmem[tid+4];
		vsmem[tid]+=vsmem[tid+2];
		vsmem[tid]+=vsmem[tid+1];

	}

	if (tid == 0)
		g_odata[blockIdx.x] = smem[0];

}
int main(int argc,char **argv)
{
    //选择gpu设备
    int dev = 0;
    if(argc > 1)
        dev = atoi(argv[1]);

    double start,end;
    cudaDeviceProp devprop;
    CHECK(cudaGetDeviceProperties(&devprop,dev));
    printf("Using device:%d : %s\n",dev,devprop.name);
    CHECK(cudaSetDevice(dev));
    //定义block大小
    int datasize = 1<<15;
    printf("datasize is :%d\n",datasize);
    int blocksize = 1024;
    dim3 block(blocksize,1);
    //定义grid大小
    // dim3 grid((datasize-1)/block.x+1,1);//reduce
    dim3 grid((datasize-1)/(block.x*2)+1,1);//reduce2 and 3
    size_t allsize = datasize*sizeof(int);
    int *host_data = (int*)malloc(allsize);
    int *host_out_data = (int *)malloc(grid.x * sizeof(int));
    for(int i=0;i<datasize;i++)
        host_data[i]=i+1;
    printf("grid %d block %d \n", grid.x, block.x);
    //计算和验证结果
    start = cpuSecond();
    int cpu_res = cpurecursiveReduce(host_data,datasize);
    end = cpuSecond();
    printf("cpu compute time is : %fms \n",end-start);

    int dev_res=0;
    printf("cpu res is:%d\n",cpu_res);
    //分配device内存
    for(int i=0;i<datasize;i++)
        host_data[i]=i+1;
    int *dev_data =NULL;
    int *dev_out_data =NULL;
    CHECK(cudaMalloc((void **)&dev_data,allsize));
    CHECK(cudaMalloc((void**)&dev_out_data,grid.x*sizeof(int)));
    CHECK(cudaMemcpy(dev_data,host_data,allsize,cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    start = cpuSecond();
    /*********dev************/
    reduce2<<<grid,block>>>(dev_data,dev_out_data,datasize);
    /**********dev*************/
    end = cpuSecond();
    printf("reduce2 gpu compute time is : %fms \n",end-start);
    // printf("dev compute finish\n");
    cudaDeviceSynchronize();
    cudaMemcpy(host_out_data,dev_out_data,grid.x*sizeof(int),cudaMemcpyDeviceToHost);
    // printf("out data is:");
    // for(int i=0;i<grid.x;i++)
    //     printf("%d\t",host_out_data[i]);
    for(int i=0;i<grid.x;i++)
        dev_res += host_out_data[i];
    printf("gpu result is :%d\n",dev_res);


    dev_res=0;
    CHECK(cudaMemcpy(dev_data,host_data,allsize,cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    start = cpuSecond();
    /*********dev************/
    reduce3<<<grid,block>>>(dev_data,dev_out_data,datasize);
    /**********dev*************/
    end = cpuSecond();
    printf("reduce4 gpu compute time is : %fms \n",end-start);
    // printf("dev compute finish\n");
    cudaDeviceSynchronize();
    cudaMemcpy(host_out_data,dev_out_data,grid.x*sizeof(int),cudaMemcpyDeviceToHost);
    // printf("out data is:");
    // for(int i=0;i<grid.x;i++)
    //     printf("%d\t",host_out_data[i]);
    for(int i=0;i<grid.x;i++)
        dev_res += host_out_data[i];
    printf("gpu result is :%d\n",dev_res);

    dev_res=0;
    CHECK(cudaMemcpy(dev_data,host_data,allsize,cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    start = cpuSecond();
    /*********dev************/
    reduce4<<<grid,block>>>(dev_data,dev_out_data,datasize);
    /**********dev*************/
    end = cpuSecond();
    printf("reduce4 gpu compute time is : %fms \n",end-start);
    // printf("dev compute finish\n");
    cudaDeviceSynchronize();
    cudaMemcpy(host_out_data,dev_out_data,grid.x*sizeof(int),cudaMemcpyDeviceToHost);
    // printf("out data is:");
    // for(int i=0;i<grid.x;i++)
    //     printf("%d\t",host_out_data[i]);
    for(int i=0;i<grid.x;i++)
        dev_res += host_out_data[i];
    printf("gpu result is :%d\n",dev_res);


    dev_res=0;
    CHECK(cudaMemcpy(dev_data,host_data,allsize,cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    start = cpuSecond();
    /*********dev************/
    // reduceSmem<<<grid.x/4,block>>>(dev_data,dev_out_data,datasize);
    reduce5<<<grid.x/4,block>>>(dev_data,dev_out_data,datasize);
    // reduceUnroll4Smem<<<grid.x/2,block>>>(dev_data,dev_out_data,datasize);
    /**********dev*************/
    end = cpuSecond();
    printf("reduce5 gpu compute time is : %fms \n",end-start);
    // printf("dev compute finish\n");
    cudaDeviceSynchronize();
    cudaMemcpy(host_out_data,dev_out_data,grid.x*sizeof(int),cudaMemcpyDeviceToHost);
    // printf("out data is:");
    // for(int i=0;i<grid.x;i++)
    //     printf("%d\t",host_out_data[i]);
    for(int i=0;i<grid.x/4;i++)
        dev_res += host_out_data[i];
    printf("gpu result is :%d\n",dev_res);
    
    free(host_data);
	free(host_out_data);
	CHECK(cudaFree(dev_data));
	CHECK(cudaFree(dev_out_data));
    cudaDeviceReset();
    return 0;
}