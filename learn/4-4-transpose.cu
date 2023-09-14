#include <cuda_runtime.h>
#include <stdio.h>
#include "util.h"

void transformMatrix2D_CPU(float * MatA,float * MatB,int nx,int ny)
{
    for(int j=0;j<ny;j++)
    {
        for(int i=0;i<nx;i++)
        {
        MatB[i*nx+j]=MatA[j*nx+i];
        }
    }
}
__global__ void copyRow(float * MatA,float * MatB,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx=ix+iy*nx;
    if (ix<nx && iy<ny)
    {
      MatB[idx]=MatA[idx];
    }
}
__global__ void copyCol(float * MatA,float * MatB,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx=ix*ny+iy;
    if (ix<nx && iy<ny)
    {
      MatB[idx]=MatA[idx];
    }
}
__global__ void transformNaiveRow(float * MatA,float * MatB,int nx,int ny)
{
    int ix = threadIdx.x+blockDim.x*blockIdx.x;
    int iy = threadIdx.y+blockDim.y*blockIdx.y;
    int idx = iy*nx+ix;
    int idy = ix*ny+iy;
    if(ix<nx && iy<ny)
        MatB[idy]=MatA[idx];
}
__global__ void transformNaiveCol(float * MatA,float * MatB,int nx,int ny)
{
    int ix = threadIdx.x+blockDim.x*blockIdx.x;
    int iy = threadIdx.y+blockDim.y*blockIdx.y;
    int idx = iy*nx+ix;
    int idy = ix*ny+iy;
    if(ix<nx && iy<ny)
        MatB[idx]=MatA[idy];
}
__global__ void transformNaiveRowUnroll(float * MatA,float * MatB,int nx,int ny)
{
    int ix = threadIdx.x+blockDim.x*blockIdx.x*4;
    int iy = threadIdx.y+blockDim.y*blockIdx.y;
    int idx = iy*nx+ix;
    int idy = ix*ny+iy;
    if(ix<nx && iy<ny)
        MatB[idy]=MatA[idx];
        MatB[idy+ny*1*blockDim.x]=MatA[idx+1*blockDim.x];
        MatB[idy+ny*2*blockDim.x]=MatA[idx+2*blockDim.x];
        MatB[idy+ny*3*blockDim.x]=MatA[idx+3*blockDim.x];
}
__global__ void transformNaiveColUnroll(float * MatA,float * MatB,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x*4;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx_row=ix+iy*nx;
    int idx_col=ix*ny+iy;
    if (ix<nx && iy<ny)
    {
        MatB[idx_row]=MatA[idx_col];
        MatB[idx_row+1*blockDim.x]=MatA[idx_col+ny*1*blockDim.x];
        MatB[idx_row+2*blockDim.x]=MatA[idx_col+ny*2*blockDim.x];
        MatB[idx_row+3*blockDim.x]=MatA[idx_col+ny*3*blockDim.x];
    }
}
__global__ void transformNaiveRowDiagonal(float * MatA,float * MatB,int nx,int ny)
{
    int block_y=blockIdx.x;
    int block_x=(blockIdx.x+blockIdx.y)%gridDim.x;
    int ix=threadIdx.x+blockDim.x*block_x;
    int iy=threadIdx.y+blockDim.y*block_y;
    int idx_row=ix+iy*nx;
    int idx_col=ix*ny+iy;
    if (ix<nx && iy<ny)
    {
      MatB[idx_col]=MatA[idx_row];
    }
}
__global__ void transformNaiveColDiagonal(float * MatA,float * MatB,int nx,int ny)
{
    int block_y=blockIdx.x;
    int block_x=(blockIdx.x+blockIdx.y)%gridDim.x;
    int ix=threadIdx.x+blockDim.x*block_x;
    int iy=threadIdx.y+blockDim.y*block_y;
    int idx_row=ix+iy*nx;
    int idx_col=ix*ny+iy;
    if (ix<nx && iy<ny)
    {
      MatB[idx_row]=MatA[idx_col];
    }
}
__global__ void transformSmem(float * in,float* out,int nx,int ny)
{
	__shared__ float tile[32][34];
	unsigned int ix,iy,transform_in_idx,transform_out_idx;
// 1
	ix=threadIdx.x+blockDim.x*blockIdx.x;
    iy=threadIdx.y+blockDim.y*blockIdx.y;
	transform_in_idx=iy*nx+ix;
// 2
	unsigned int bidx,irow,icol;
	bidx=threadIdx.y*blockDim.x+threadIdx.x;
	irow=bidx/blockDim.y;
	icol=bidx%blockDim.y;
// 3
	ix=blockIdx.y*blockDim.y+icol;
	iy=blockIdx.x*blockDim.x+irow;
// 4
	transform_out_idx=iy*ny+ix;
	if(ix<nx&& iy<ny)
	{
		tile[threadIdx.y][threadIdx.x]=in[transform_in_idx];
		__syncthreads();
		out[transform_out_idx]=tile[icol][irow];

	}
}
int main(int argc,char **argv)
{
    //选择gpu设备
    int dev = 0;
    if(argc > 1)
        dev = atoi(argv[1]);

    cudaDeviceProp devprop;
    //设置设备
    CHECK(cudaGetDeviceProperties(&devprop,dev));
    printf("Using device:%d : %s\n",dev,devprop.name);
    CHECK(cudaSetDevice(dev));

    int transform_kernel=0;
    if(argc>=3)
        transform_kernel=atoi(argv[2]);
    //设置转置矩阵大小
    int nx=1<<12;
    int ny=1<<12;
    int nxy=nx*ny;
    int nBytes=nxy*sizeof(float);
    //Malloc
    float* A_host=(float*)malloc(nBytes);
    float* B_host=(float*)malloc(nBytes);
    initialData(A_host,nxy);

    //cudaMalloc
    float *A_dev=NULL;
    float *B_dev=NULL;
    CHECK(cudaMalloc((void**)&A_dev,nBytes));
    CHECK(cudaMalloc((void**)&B_dev,nBytes));

    CHECK(cudaMemcpy(A_dev,A_host,nBytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemset(B_dev,0,nBytes));

    int dimx=32;
    int dimy=32;

    // cpu compute
    double iStart=cpuSecond();
    transformMatrix2D_CPU(A_host,B_host,nx,ny);
    double iElaps=cpuSecond()-iStart;
    printf("CPU Execution Time elapsed %f sec\n",iElaps);
    // 2d block and 2d grid
    dim3 block(dimx,dimy);
    dim3 grid((nx-1)/block.x+1,(ny-1)/block.y+1);
    dim3 block_1(dimx,dimy);
    dim3 grid_1((nx-1)/(block_1.x*4)+1,(ny-1)/block_1.y+1);
    iStart=cpuSecond();
    switch(transform_kernel)
    {
    case 0:
        copyRow<<<grid,block>>>(A_dev,B_dev,nx,ny);
        break;
    case 1:
        copyCol<<<grid,block>>>(A_dev,B_dev,nx,ny);
        break;
    case 2:
        transformNaiveRow<<<grid,block>>>(A_dev,B_dev,nx,ny);
        break;
    case 3:
        transformNaiveCol<<<grid,block>>>(A_dev,B_dev,nx,ny);
        break;
    case 4:
        transformNaiveRowUnroll<<<grid_1,block_1>>>(A_dev,B_dev,nx,ny);
            break;
    case 5:
        transformNaiveColUnroll<<<grid_1,block_1>>>(A_dev,B_dev,nx,ny);
            break;
    case 6:
        transformNaiveRowDiagonal<<<grid,block>>>(A_dev,B_dev,nx,ny);
            break;
    case 7:
        transformNaiveColDiagonal<<<grid,block>>>(A_dev,B_dev,nx,ny);
            break;
    case 8:
        transformSmem<<<grid,block>>>(A_dev,B_dev,nx,ny);
            break;
    default:
        break;
    }
    CHECK(cudaDeviceSynchronize());
    iElaps=cpuSecond()-iStart;
    printf(" Time elapsed %f sec\n",iElaps);
    CHECK(cudaMemcpy(B_host,B_dev,nBytes,cudaMemcpyDeviceToHost));
    checkResult(B_host,B_host,nxy);

    cudaFree(A_dev);
    cudaFree(B_dev);
    free(A_host);
    free(B_host);
    cudaDeviceReset();
    return 0;
}