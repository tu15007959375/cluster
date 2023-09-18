#include <cuda_runtime.h>
#include "mmio.h"
#include "util.h"
#include<iostream>
#include<map>
#include"omp.h"
#include<algorithm>
#include<vector>
#include<set>
#include <limits.h>
#include<numeric>
using namespace std;
#define CHECK_RESULT//计算cpu的结果并且与gpu的结果进行验证
#define COMPUTE_GFLOPS//通过迭代次数进行计算并且输出gfloats
// #define DISPLAY_BDIA_INFO
// #define DISPLAY_ROW_BEGIN
// CUDA内核：DIA格式的SPMV
__global__ void spmv_kernel(const struct DIAFormat* dia_matrix, const float* vector, float* result, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < (num_rows)) {
        float value = 0;
        for(int i =0; i < dia_matrix->num_diagonals; i++)
        {
            int index = dia_matrix->diagonals_offsets[i] + row;
            if(index<num_rows&&index>=0)
            {
                value += dia_matrix->diagonals_values[(long)row+dia_matrix->max_diag_length*i] * vector[index];
            }
            
        }
        result[row] = value;
    }
    
}

// CUDA内核：BRCSD2格式的SPMV
__global__ void brcsd2_spmv_kernel(const struct BRCSD2* brcsd2, const float* vector, float* result, int num_rows, int per_block_tile) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float res = 0;
    if(row>=num_rows)
        return;
    int tile_row=row+1;
    int index = -1;//索引
    int offset = 0;//前对角线数目
    int offset_val_index=0;//当前块对角线值的索引
    //寻找第几块
    while(tile_row>0)
    {
        index++;
        tile_row -= brcsd2->offset_key[index]*brcsd2->nrows;
        offset += brcsd2->offset_key[index]*brcsd2->val_size[index];
        offset_val_index += brcsd2->val_size[index];
    }
    tile_row += brcsd2->offset_key[index]*brcsd2->nrows;
    tile_row --;//当前第几大块
    offset_val_index -=  brcsd2->val_size[index];
    int inner_id = tile_row/brcsd2->nrows;//当前大块中的第几小块
    tile_row = tile_row%brcsd2->nrows;//小块中的第几行
    int nowdiags = brcsd2->val_size[index];//当前块的对角线数量
    offset -= brcsd2->offset_key[index]*nowdiags;//大块之前的对角线总数量
    int data_index = (offset+inner_id*nowdiags)*brcsd2->nrows+tile_row;
    int tmp = 0;
    for(int i =0;i<nowdiags;i++)
    {
        // printf("id:%d,data_index:%d\n",row,data_index+i*brcsd2->nrows);
        tmp = brcsd2->offset_val[i+offset_val_index]+row;
        if(tmp>=0)
            res += brcsd2->data[data_index+i*brcsd2->nrows]*vector[tmp];
        // res += brcsd2->data[data_index+i*brcsd2->nrows]*vector[abs(brcsd2->offset_val[i+offset_val_index]+row)];
    }
    result[row] = res;
    // printf("id:%d,inner_id:%d,tile_row:%d,offset:%d,index:%d\n",row,inner_id,tile_row,offset,index);
}

// CUDA内核：BRCSD2_2格式的SPMV
__global__ void brcsd2_spmv_kernel_2(const struct BRCSD2* brcsd2, const float* vector, float* result, int num_rows, int per_block_tile) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float res = 0;
    if(row>=num_rows)
        return;
    int tile_row=row+1;
    int index = -1;//索引
    int offset = 0;//前对角线数目
    int offset_val_index=0;//当前块对角线值的索引
    //寻找第几块
    while(tile_row>0)
    {
        index++;
        tile_row -= brcsd2->offset_key[index]*brcsd2->nrows;
        offset += brcsd2->offset_key[index]*brcsd2->val_size[index];
        offset_val_index += brcsd2->val_size[index];
    }
    tile_row += brcsd2->offset_key[index]*brcsd2->nrows;
    tile_row --;//当前第几大块
    offset_val_index -=  brcsd2->val_size[index];
    int inner_id = tile_row/brcsd2->nrows;//当前大块中的第几小块
    tile_row = tile_row%brcsd2->nrows;//小块中的第几行
    int nowdiags = brcsd2->val_size[index];//当前块的对角线数量
    offset -= brcsd2->offset_key[index]*nowdiags;//大块之前的对角线总数量
    int data_index = (offset+inner_id*nowdiags)*brcsd2->nrows+tile_row*nowdiags;
    int tmp = 0;
    for(int i =0;i<nowdiags;i++)
    {
        // printf("id:%d,data_index:%d\n",row,data_index+i*brcsd2->nrows);
        tmp = brcsd2->offset_val[i+offset_val_index]+row;
        if(tmp>=0)
        {
            // printf("row:%d,%f\n",row,brcsd2->data[data_index+i]);
            res += brcsd2->data[data_index+i]*vector[0];
        }
        // res += brcsd2->data[data_index+i*brcsd2->nrows]*vector[0];
    }
    result[row] = res;
    // printf("id:%d,inner_id:%d,tile_row:%d,offset:%d,index:%d\n",row,inner_id,tile_row,offset,index);
}

// CUDA内核：BDIA格式的SPMV
__global__ void bdia_spmv_kernel(const struct BDIA* bdia,float* vector, float* result, int m) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    // if(row==0)
    if(row<m)
    {
        float res = 0;
        int tile_index = bdia->row_tile_index[row];//处于第几块
        int index = bdia->data_begin[tile_index];//前n块的数据数量
        int rows = bdia->row_begin[tile_index+1]-bdia->row_begin[tile_index];//该块每条对角线长度
        int nowrow = row - bdia->row_begin[tile_index];
        // int rows = row_begin[tile_index+1]-row_begin[tile_index];//该块每条对角线长度
        // int nowrow = row - row_begin[tile_index];//位于该块第几行
        index = index + nowrow;
        int for_end = bdia->val_size[tile_index];
        int dia_offset = bdia->dia_begin[tile_index];
        int *offset_val = bdia->offset_val+dia_offset;
        // int col_index = *(offset_val)+row;
        float *data_offset = bdia->data+index;
        for(int i =0;i<for_end;i++)
        {
            // int offset = *(offset_val+i)-*(offset_val);
            // res += (*(data_offset+i*rows))*vector[abs(col_index+offset)];
            int col_index = *(offset_val+i)+row;
            res += (*(data_offset+i*rows))*vector[abs(col_index)];
        }
        result[row] = res;
    }
    
}

// CUDA内核：带share memory
__global__ void bdia_spmv_kernel_with_share(const struct BDIA* bdia,float* vector, float* result, int m,int allmin,int allmax,int share_size)
{
    //share
    extern __shared__ float x[];
    int tid = threadIdx.x;
    int index = tid;
    int min_offset = allmin+blockIdx.x*1024;
    // int max_offset =min_offset+share_size;
    // int elements_per_thread = (share_size-1) / blockDim.x+1; 
    while (index < share_size) 
    {
        // 在这里赋值共享内存
        x[index] = vector[min_offset+index]; // 您的赋值操作
        index += blockDim.x; // 移动到下一个线程负责的索引
    }
    __syncthreads();
    //share

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row<m)
    {
        float res = 0;
        int tile_index = bdia->row_tile_index[row];//处于第几块
        int index = bdia->data_begin[tile_index];//前n块的数据数量
        int rows = bdia->row_begin[tile_index+1]-bdia->row_begin[tile_index];//该块每条对角线长度
        int nowrow = row - bdia->row_begin[tile_index];
        // int rows = row_begin[tile_index+1]-row_begin[tile_index];//该块每条对角线长度
        // int nowrow = row - row_begin[tile_index];//位于该块第几行
        index = index + nowrow;
        int for_end = bdia->val_size[tile_index];
        int dia_offset = bdia->dia_begin[tile_index];
        int *offset_val = bdia->offset_val+dia_offset;
        // int col_index = *(offset_val)+row;
        float *data_offset = bdia->data+index;
        for(int i =0;i<for_end;i++)
        {
            // int offset = *(offset_val+i)-*(offset_val);
            // res += (*(data_offset+i*rows))*vector[abs(col_index+offset)];
            int col_index = *(offset_val+i)+row-min_offset;
            res += (*(data_offset+i*rows))*x[abs(col_index)];
        }
        result[row] = res;
    }
}

// COO格式到DIA格式的转换函数
void convertCOOtoDIA(const struct COOElement* coo_elements, int num_elements, int num_rows, int num_cols, struct DIAFormat* dia_matrix) {
    // 计算对角线的数量和最大对角线长度
    int max_offset = (num_rows > num_cols) ? num_rows : num_cols;
    int* diag_count = (int*)calloc(2 * max_offset - 1, sizeof(int));
    // #pragma omp parallel for
    for (int i = 0; i < num_elements; ++i) {
        int offset = coo_elements[i].col - coo_elements[i].row + max_offset - 1;
        diag_count[offset]++;
    }

    // 分配内存以存储对角线偏移和值
    int num_diagonals = 0;
    // #pragma omp parallel for
    for (int i = 0; i < 2 * max_offset - 1; ++i) {
        if (diag_count[i] > 0) {
            num_diagonals++;
        }
    }
    dia_matrix->num_diagonals = num_diagonals;
    dia_matrix->max_diag_length = max_offset;

    dia_matrix->diagonals_offsets = (int*)malloc(num_diagonals * sizeof(int));
    dia_matrix->diagonals_values = (float*)malloc((long)num_diagonals * max_offset * sizeof(float));

    int dia_idx = 0;
    // printf("max_offset%d,num_diagonals%d",max_offset,num_diagonals);
    memset(dia_matrix->diagonals_values,0,(long)num_diagonals * max_offset*sizeof(float));
    // #pragma omp parallel for
    for (int i = 0; i < 2 * max_offset - 1; ++i) {
        if (diag_count[i] > 0) {
            //计算一根对角线相对主对角线的偏移值
            dia_matrix->diagonals_offsets[dia_idx] = i - max_offset + 1;
            dia_idx++;
        }
    }
    // #pragma omp parallel for
    for (int j = 0; j < num_elements; ++j) 
    {
        int row = coo_elements[j].row;
        int col = coo_elements[j].col;
        float value = coo_elements[j].value;
        int offset = col-row;
        //二分查找
        int pos = (int*) bsearch (&offset, dia_matrix->diagonals_offsets, dia_matrix->num_diagonals, sizeof (int), cmpfunc)-dia_matrix->diagonals_offsets;

        *(dia_matrix->diagonals_values+(long)pos * max_offset + row) = value;

    }

    free(diag_count);
}

void spmv_cpu(DIAFormat* dia_matrix, const float* vector, float* result, int num_rows) {

    // #pragma omp parallel for
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < dia_matrix->num_diagonals; ++j) {
            int offset = dia_matrix->diagonals_offsets[j];
            float value = dia_matrix->diagonals_values[(long)j * dia_matrix->max_diag_length + i];
            int col_idx = i + offset;
            if (col_idx >= 0 && col_idx < num_rows) {
                result[i] += value * vector[col_idx];
            }
        }
    }
}

void gpu_test_dia(struct DIAFormat* dia_matrix,float *x,int m,int ne,int timing_iterations,float* result_cpu)
{
    struct DIAFormat* dia_matrix_d;
    float* vector_d;
    float* result_d;

    int *diagonals_offsets_d;
    float *diagonals_values_d;
    printf("dia零元填充:%d\n",m*dia_matrix->num_diagonals-ne);
    cudaMalloc((void**)&diagonals_offsets_d, sizeof(int)*dia_matrix->num_diagonals);
    cudaMalloc((void**)&diagonals_values_d, sizeof(float)*m*dia_matrix->num_diagonals);
    cudaMemcpy(diagonals_offsets_d, dia_matrix->diagonals_offsets, sizeof(int)*dia_matrix->num_diagonals, cudaMemcpyHostToDevice);
    cudaMemcpy(diagonals_values_d, dia_matrix->diagonals_values, sizeof(float)*m*dia_matrix->num_diagonals, cudaMemcpyHostToDevice);
    int *dia_matrix_tmp1 = dia_matrix->diagonals_offsets;
    float *dia_matrix_tmp2 = dia_matrix->diagonals_values;
    dia_matrix->diagonals_offsets = diagonals_offsets_d;
    dia_matrix->diagonals_values = diagonals_values_d;

    cudaMalloc((void**)&dia_matrix_d, sizeof(struct DIAFormat));
    cudaMemcpy(dia_matrix_d, dia_matrix, sizeof(struct DIAFormat), cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&vector_d, sizeof(float) * m);
    cudaMemcpy(vector_d, x, sizeof(float) * m, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&result_d, sizeof(float) * m);

    // 启动 CUDA 内核,唤醒一次
    int block_size = 1024;
    int num_blocks = (m + block_size - 1) / block_size;
    spmv_kernel<<<num_blocks, block_size>>>(dia_matrix_d, vector_d, result_d, m);
    #ifdef COMPUTE_GFLOPS
        float time_elapsed=0;
        cudaEvent_t start,stop;
        cudaEventCreate(&start);    //创建Event
        cudaEventCreate(&stop);
        cudaEventRecord(start,0);
        for(int i = 0;i<timing_iterations; i++)
        {
            spmv_kernel<<<num_blocks, block_size>>>(dia_matrix_d, vector_d, result_d, m);
        }
        cudaEventRecord( stop,0);
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_elapsed,start,stop);
        time_elapsed /= 1000.0 * (float) timing_iterations;

        double gflops = 2*(ne);
        gflops /=time_elapsed * 1000000000;
        printf("dia gflops:%f\n",gflops);
    #endif
    dia_matrix->diagonals_offsets = dia_matrix_tmp1;
    dia_matrix->diagonals_values = dia_matrix_tmp2;
    // 从 GPU 获取结果,打印结果
    // for(int i =0;i<m;i++)
    // {
    //     printf("%f\n",result_h[i]);
    // }
    #ifdef CHECK_RESULT
        float *result_h = (float*)malloc(sizeof(float) * m);
        cudaMemcpy(result_h, result_d, sizeof(float) * m, cudaMemcpyDeviceToHost);
        checkResult(result_cpu,result_h,m);
    #endif
}

void gpu_test_brcsd2(struct DIAFormat* dia_matrix,float *x,int m,int ne,int timing_iterations,float* result_cpu,int nrows)
{
    int tile_size = (m-1)/nrows+1;

    //初始化
    struct BRCSD2 *brcsd2;
    brcsd2 = (struct BRCSD2*)malloc(sizeof(struct BRCSD2));
    brcsd2->tile_size = tile_size;
    brcsd2->ac_size = tile_size;
    brcsd2->nrows = nrows;
    brcsd2->offset_key = (int *)malloc(sizeof(int)*tile_size);
    brcsd2->val_size = (int *)malloc(sizeof(int)*tile_size);
    int max_diag_length = dia_matrix->max_diag_length;
    // #pragma omp parallel for
    for(int i = 0;i<tile_size;i++)
        brcsd2->offset_key[i]=1;

    //计算val_size,
    // int max_index = max_diag_length*dia_matrix->num_diagonals;
    // printf("max_index,%d",max_index);
    // #pragma omp parallel for
    for(int i = 0;i<tile_size;i++)
    {
        int count = 0;
        for(int j =0;j<dia_matrix->num_diagonals;j++)
        {
            bool flag = is_all_zero(dia_matrix->diagonals_values+max_diag_length*j,i*nrows,MIN((i+1)*nrows,max_diag_length));
            // printf("index:%d,(i+1)*nrows:%d,%d\n",max_diag_length*j,(i)*nrows,*(dia_matrix->diagonals_values+max_index));
            if(!flag)
            {
                count++;
            }
        }
        brcsd2->val_size[i]=count;
    }

    //分配offset_val和data空间

    int all_dia_num=0;
    for(int i = 0;i<tile_size;i++)
    {
        all_dia_num += brcsd2->val_size[i];
    }
    brcsd2->offset_val = (int*)malloc(sizeof(int)*all_dia_num);
    brcsd2->data = (float *)malloc(sizeof(float)*all_dia_num*nrows);
    memset(brcsd2->data,0,sizeof(float)*all_dia_num*nrows);
    //赋值dia和data
    int start=0;
    int count2 = 0;
    // #pragma omp parallel for
    for(int i = 0;i<tile_size;i++)
    {
        int count = 0;
        
        for(int j =0;j<dia_matrix->num_diagonals;j++)
        {
            bool flag = is_all_zero(dia_matrix->diagonals_values+max_diag_length*j,i*nrows,MIN((i+1)*nrows,max_diag_length));
            if(!flag)
            {
                //赋值每块对角线的偏移值
                *(brcsd2->offset_val+start+count) = dia_matrix->diagonals_offsets[j];
                for(int k = 0;k < nrows; k++)
                {
                    if(k+i*nrows>=max_diag_length)
                    {
                        // printf("k+i*nrows:%d\t",k+i*nrows);
                        *(brcsd2->data+count2) = 0;
                    }
                    else
                        *(brcsd2->data+count2) = *(dia_matrix->diagonals_values+max_diag_length*j+k+i*nrows);
                    // printf("test:val:%f ",*(dia_matrix->diagonals_values+max_diag_length*j+k+i*nrows));
                    count2++;
                }
                count++;
            }
        }
        start += brcsd2->val_size[i];
    }
    // display_brcsd2_info(brcsd2);
    //合并过程
    start = 0;
    int start2 =0;
    int ii =0;
    while(ii<tile_size-1)
    {
        start2 = start+brcsd2->val_size[ii];
        // printf("ii:%d,start:%d,start2:%d,brcsd2->val_size:%d\n",ii,start,start2,brcsd2->val_size[ii]);
        if((brcsd2->val_size[ii]==brcsd2->val_size[ii+1])&&memcmp(brcsd2->offset_val+start,brcsd2->offset_val+start2,brcsd2->val_size[ii])==0)
        {
            brcsd2->offset_key[ii]++;
            del_index_num(brcsd2->offset_key,brcsd2->ac_size,ii+1);
            del_index_num2(brcsd2->offset_val,all_dia_num,start2,brcsd2->val_size[ii+1]);
            del_index_num(brcsd2->val_size,brcsd2->ac_size,ii+1);
            brcsd2->ac_size--;
            all_dia_num = all_dia_num-brcsd2->val_size[ii+1];
            tile_size--;
        }
        else
        {
            start += brcsd2->val_size[ii];
            ii++;
        }
    }

    // display_brcsd2_info(brcsd2,all_dia_num);

    all_dia_num=0;
    int all_data_num=0;//为总对角线数量
    for(int i = 0;i<brcsd2->ac_size;i++)
    {
        all_dia_num += brcsd2->val_size[i];
        all_data_num += brcsd2->val_size[i]*brcsd2->offset_key[i];
    }
    // printf("all_dia_num:%d,all_data_num:%d\n",all_dia_num,all_data_num);
    // printf("count2:%d\n",count2);
    printf("brcsd2零元填充:%d\n",all_data_num*brcsd2->nrows-ne);
    //device定义

    struct BRCSD2 *brcsd2_d;
    int *offset_key_d,*val_size_d;
    int *offset_val_d;
    float *data_d;
    float* vector_d;
    float* result_d;

    cudaMalloc((void**)&offset_key_d, sizeof(int)*brcsd2->ac_size);
    cudaMalloc((void**)&val_size_d, sizeof(int)*brcsd2->ac_size);
    cudaMalloc((void**)&offset_val_d, sizeof(int)*all_dia_num);
    cudaMalloc((void**)&data_d, sizeof(float)*all_data_num*brcsd2->nrows);
    cudaMemcpy(offset_key_d, brcsd2->offset_key, sizeof(int)*brcsd2->ac_size, cudaMemcpyHostToDevice);
    cudaMemcpy(val_size_d, brcsd2->val_size, sizeof(int)*brcsd2->ac_size, cudaMemcpyHostToDevice);
    cudaMemcpy(offset_val_d, brcsd2->offset_val, sizeof(int)*all_dia_num, cudaMemcpyHostToDevice);
    cudaMemcpy(data_d, brcsd2->data, sizeof(float)*all_data_num*brcsd2->nrows, cudaMemcpyHostToDevice);

    brcsd2->offset_key=offset_key_d;
    brcsd2->val_size=val_size_d;
    brcsd2->offset_val=offset_val_d;
    brcsd2->data=data_d;

    cudaMalloc((void**)&brcsd2_d, sizeof(struct BRCSD2));
    cudaMemcpy(brcsd2_d, brcsd2, sizeof(struct BRCSD2), cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&vector_d, sizeof(float) * m);
    cudaMemcpy(vector_d, x, sizeof(float) * m, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&result_d, sizeof(float) * m);

    // 启动 CUDA 内核,唤醒一次

    int block_size = 1024;
    int num_blocks = (m - 1) / block_size + 1;
    int per_block_tile = block_size/brcsd2->nrows;
    // printf("per_block_tile:%d\n",per_block_tile);
    brcsd2_spmv_kernel<<<num_blocks, block_size>>>(brcsd2_d, vector_d, result_d, m, per_block_tile);
    #ifdef COMPUTE_GFLOPS
        float time_elapsed=0;
        cudaEvent_t starten,stop;
        cudaEventCreate(&starten);    //创建Event
        cudaEventCreate(&stop);
        cudaEventRecord(starten,0);
        for(int i = 0;i<timing_iterations; i++)
        {
            brcsd2_spmv_kernel<<<num_blocks, block_size>>>(brcsd2_d, vector_d, result_d, m, per_block_tile);
        }
        cudaEventRecord(stop,0);
        cudaEventSynchronize(starten);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_elapsed,starten,stop);
        time_elapsed /= 1000.0 * (float) timing_iterations;
        double gflops = 2*(ne);
        gflops /=time_elapsed * 1000000000;
        printf("brcsd2 gflops:%f\n",gflops);
    #endif
    //该处lightspmv是2*（m+ne），而merge是2*ne



    // 从 GPU 获取结果,打印结果
    #ifdef CHECK_RESULT
        float *result_h = (float*)malloc(sizeof(float) * m);
        cudaMemcpy(result_h, result_d, sizeof(float) * m, cudaMemcpyDeviceToHost);
        checkResult(result_cpu,result_h,m);
    #endif

    
    // // printf("gpu Result after SPMV:\n");
    // for (int i = 0; i < m; ++i) {
    //     printf("%f\n", result_h[i]);
    // }
    // printf("\n");
    
}

void gpu_test_brcsd2_2(struct DIAFormat* dia_matrix,float *x,int m,int ne,int timing_iterations,float* result_cpu,int nrows)
{
    int tile_size = (m-1)/nrows+1;

    //初始化
    struct BRCSD2 *brcsd2;
    brcsd2 = (struct BRCSD2*)malloc(sizeof(struct BRCSD2));
    brcsd2->tile_size = tile_size;
    brcsd2->ac_size = tile_size;
    brcsd2->nrows = nrows;
    brcsd2->offset_key = (int *)malloc(sizeof(int)*tile_size);
    brcsd2->val_size = (int *)malloc(sizeof(int)*tile_size);
    int max_diag_length = dia_matrix->max_diag_length;
    for(int i = 0;i<tile_size;i++)
        brcsd2->offset_key[i]=1;

    //计算val_size,
    // int max_index = max_diag_length*dia_matrix->num_diagonals;
    // printf("max_index,%d",max_index);
    for(int i = 0;i<tile_size;i++)
    {
        int count = 0;
        for(int j =0;j<dia_matrix->num_diagonals;j++)
        {
            bool flag = is_all_zero(dia_matrix->diagonals_values+max_diag_length*j,i*nrows,MIN((i+1)*nrows,max_diag_length));
            // printf("index:%d,(i+1)*nrows:%d,%d\n",max_diag_length*j,(i)*nrows,*(dia_matrix->diagonals_values+max_index));
            if(!flag)
            {
                count++;
            }
        }
        brcsd2->val_size[i]=count;
    }

    //分配offset_val和data空间

    int all_dia_num=0;
    for(int i = 0;i<tile_size;i++)
    {
        all_dia_num += brcsd2->val_size[i];
    }
    brcsd2->offset_val = (int*)malloc(sizeof(int)*all_dia_num);
    brcsd2->data = (float *)malloc(sizeof(float)*all_dia_num*nrows);
    memset(brcsd2->data,0,sizeof(float)*all_dia_num*nrows);
    //赋值dia和data

    int start=0;
    int count2 = 0;
    for(int i = 0;i<tile_size;i++)
    {
        int count = 0;
        
        for(int j =0;j<dia_matrix->num_diagonals;j++)
        {
            bool flag = is_all_zero(dia_matrix->diagonals_values+max_diag_length*j,i*nrows,MIN((i+1)*nrows,max_diag_length));
            if(!flag)
            {
                count2 = count+start*brcsd2->nrows;
                //赋值每块对角线的偏移值
                *(brcsd2->offset_val+start+count) = dia_matrix->diagonals_offsets[j];
                for(int k = 0;k < nrows; k++)
                {
                    if(k+i*nrows<max_diag_length)
                    {
                        // printf("count2:%d\n",count2);
                        *(brcsd2->data+count2) = *(dia_matrix->diagonals_values+max_diag_length*j+k+i*nrows);
                    }           
                    // printf("test:val:%f ",*(dia_matrix->diagonals_values+max_diag_length*j+k+i*nrows));
                    count2 += brcsd2->val_size[i];
                }
                count++;
            }
        }
        start += brcsd2->val_size[i];
    }
    // display_brcsd2_info(brcsd2);
    //合并过程
    start = 0;
    int start2 =0;
    int ii =0;
    while(ii<tile_size-1)
    {
        start2 = start+brcsd2->val_size[ii];
        // printf("ii:%d,start:%d,start2:%d,brcsd2->val_size:%d\n",ii,start,start2,brcsd2->val_size[ii]);
        if((brcsd2->val_size[ii]==brcsd2->val_size[ii+1])&&memcmp(brcsd2->offset_val+start,brcsd2->offset_val+start2,brcsd2->val_size[ii])==0)
        {
            brcsd2->offset_key[ii]++;
            del_index_num(brcsd2->offset_key,brcsd2->ac_size,ii+1);
            del_index_num2(brcsd2->offset_val,all_dia_num,start2,brcsd2->val_size[ii+1]);
            del_index_num(brcsd2->val_size,brcsd2->ac_size,ii+1);
            brcsd2->ac_size--;
            all_dia_num = all_dia_num-brcsd2->val_size[ii+1];
            tile_size--;
        }
        else
        {
            start += brcsd2->val_size[ii];
            ii++;
        }
    }


    all_dia_num=0;
    int all_data_num=0;
    for(int i = 0;i<brcsd2->ac_size;i++)
    {
        all_dia_num += brcsd2->val_size[i];
        all_data_num += brcsd2->val_size[i]*brcsd2->offset_key[i];
    }
    // display_brcsd2_info(brcsd2,all_data_num);
    // printf("all_dia_num:%d,all_data_num:%d\n",all_dia_num,all_data_num);
    // printf("count2:%d\n",count2);
    //device定义

    struct BRCSD2 *brcsd2_d;
    int *offset_key_d,*val_size_d;
    int *offset_val_d;
    float *data_d;
    float* vector_d;
    float* result_d;

    cudaMalloc((void**)&offset_key_d, sizeof(int)*brcsd2->ac_size);
    cudaMalloc((void**)&val_size_d, sizeof(int)*brcsd2->ac_size);
    cudaMalloc((void**)&offset_val_d, sizeof(int)*all_dia_num);
    cudaMalloc((void**)&data_d, sizeof(float)*all_data_num*brcsd2->nrows);
    cudaMemcpy(offset_key_d, brcsd2->offset_key, sizeof(int)*brcsd2->ac_size, cudaMemcpyHostToDevice);
    cudaMemcpy(val_size_d, brcsd2->val_size, sizeof(int)*brcsd2->ac_size, cudaMemcpyHostToDevice);
    cudaMemcpy(offset_val_d, brcsd2->offset_val, sizeof(int)*all_dia_num, cudaMemcpyHostToDevice);
    cudaMemcpy(data_d, brcsd2->data, sizeof(float)*all_data_num*brcsd2->nrows, cudaMemcpyHostToDevice);

    brcsd2->offset_key=offset_key_d;
    brcsd2->val_size=val_size_d;
    brcsd2->offset_val=offset_val_d;
    brcsd2->data=data_d;

    cudaMalloc((void**)&brcsd2_d, sizeof(struct BRCSD2));
    cudaMemcpy(brcsd2_d, brcsd2, sizeof(struct BRCSD2), cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&vector_d, sizeof(float) * m);
    cudaMemcpy(vector_d, x, sizeof(float) * m, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&result_d, sizeof(float) * m);

    // 启动 CUDA 内核,唤醒一次

    int block_size = 1024;
    int num_blocks = (m - 1) / block_size + 1;
    int per_block_tile = block_size/brcsd2->nrows;
    // printf("per_block_tile:%d\n",per_block_tile);
    brcsd2_spmv_kernel_2<<<num_blocks, block_size>>>(brcsd2_d, vector_d, result_d, m, per_block_tile);
    #ifdef COMPUTE_GFLOPS
        float time_elapsed=0;
        cudaEvent_t starten,stop;
        cudaEventCreate(&starten);    //创建Event
        cudaEventCreate(&stop);
        cudaEventRecord(starten,0);
        for(int i = 0;i<timing_iterations; i++)
        {
            brcsd2_spmv_kernel_2<<<num_blocks, block_size>>>(brcsd2_d, vector_d, result_d, m, per_block_tile);
        }
        cudaEventRecord(stop,0);
        cudaEventSynchronize(starten);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_elapsed,starten,stop);
        time_elapsed /= 1000.0 * (float) timing_iterations;
        double gflops = 2*(ne);
        printf("brcsd2_2 gflops:%f\n",gflops);

    //该处lightspmv是2*（m+ne），而merge是2*ne
    #endif


    // 从 GPU 获取结果,打印结果
    #ifdef CHECK_RESULT
        float *result_h = (float*)malloc(sizeof(float) * m);
        cudaMemcpy(result_h, result_d, sizeof(float) * m, cudaMemcpyDeviceToHost);
        checkResult(result_cpu,result_h,m);
    #endif

    
    // // printf("gpu Result after SPMV:\n");
    // for (int i = 0; i < m; ++i) {
    //     printf("%f\n", result_h[i]);
    // }
    // printf("\n");
    
}


void gpu_test_bdia(struct COOElement* coo_elements,int m,int ne,int timing_iterations,float* result_cpu,int maxdiff)
{
    vector<int> rowbeginvec;//每块的行偏移
    vector<vector<int>> diaoffsetvec;//存放每一行的对角线偏移值
    vector<vector<int>> diaoffsetvecdis;//每块的对角线偏移值
    vector<long> databegin;//每块数据的开始偏移
    map<int,vector<int>> diamap;//存放行：偏移值的键值对
    vector<int> tmpvec;
    struct BDIA *bdia = (struct BDIA *)malloc(sizeof(struct BDIA));
    int tilesize=0;
    long alldatasize=0;
    // int maxdiff = 3;
    int maxtilesize = 512;
    int mintilesize = 32;//32
    int alldianums = 0;

    // #pragma omp parallel for
    // #pragma omp parallel
    for(int i=0;i<ne;i++)
    {
        diamap[coo_elements[i].row].emplace_back(coo_elements[i].col-coo_elements[i].row);
        
    }

    for(auto it:diamap)
    {
        diaoffsetvec.emplace_back(it.second);

        // cout<<it.first<<":";
        // cout<<it.second.size();
        // cout<<endl;
    }
    //share memory
    int xtestcount = 0;
    int xmin = INT_MAX;
    int xmax = INT_MIN;
    int allmin = 0;
    int allmax = 0;
    for(int i = 0;i<diaoffsetvec.size();i++)
    {
        if(xmin>diaoffsetvec[i][0])
            xmin = diaoffsetvec[i][0];
        if(xmax<diaoffsetvec[i].back())
            xmax = diaoffsetvec[i].back();
        xtestcount++;
        // printf("xtestcount:%d\n",xtestcount);
        if(xtestcount==1024)
        {
            if(xmin<allmin)
                allmin = xmin;
            if(xmax>allmax)
                allmax = xmax;
            xmin = INT_MAX;
            xmax = INT_MIN;
            xtestcount=0;
        }
    }
    // printf("allmin:%d,allmax:%d\n",allmin,allmax);
    //share memory
    int pertile_min_len=0;
    int pertile_max_len=0;
    rowbeginvec.emplace_back(0);
    tmpvec = diaoffsetvec[0];
    int tilecount = 0;
    // #pragma omp parallel for
    for(int i=0;i<diaoffsetvec.size();i++)
    {
        if(diaoffsetvec[i].size()>diaoffsetvec[pertile_max_len].size())
            pertile_max_len = i;
        if(diaoffsetvec[i].size()<diaoffsetvec[pertile_min_len].size())
            pertile_min_len = i;
        int tmp = tmp = diaoffsetvec[pertile_max_len].size()-diaoffsetvec[pertile_min_len].size();
        if((tmp>maxdiff||tilecount>maxtilesize)&&tilecount>=mintilesize)
        {
            rowbeginvec.emplace_back(i);
            diaoffsetvecdis.emplace_back(tmpvec);
            vector <int>().swap(tmpvec); 
            tmpvec.insert(tmpvec.end(),diaoffsetvec[i].begin(),diaoffsetvec[i].end());
            pertile_min_len = i;
            pertile_max_len = i;
            tilecount = 0;
        }
        else
        {
            tmpvec.insert(tmpvec.end(),diaoffsetvec[i].begin(),diaoffsetvec[i].end());
            set<int> st(tmpvec.begin(),tmpvec.end());
            tmpvec.assign(st.begin(), st.end());
            tilecount++;
        }

    }
    diaoffsetvecdis.emplace_back(tmpvec);

    //将数据存入结构体
    tilesize = rowbeginvec.size();
    rowbeginvec.emplace_back(m);
    bdia->tile_size = tilesize;
    bdia->row_begin = (int *)malloc(sizeof(int)*(tilesize+1));
    bdia->val_size = (int *)malloc(sizeof(int)*tilesize);
    bdia->data_begin = (long *)malloc(sizeof(long)*tilesize);
    bdia->dia_begin = (int *)malloc(sizeof(int)*tilesize);
    bdia->row_tile_index = (int *)malloc(sizeof(int)*m);
    // bdia->for_begin = (int *)malloc(sizeof(int)*2*m);
    for(int i =0;i<tilesize;i++)
    {
        bdia->row_begin[i] = rowbeginvec[i];
        bdia->val_size[i] = diaoffsetvecdis[i].size();
        alldianums += diaoffsetvecdis[i].size();
    }
    bdia->row_begin[tilesize] = m;
    databegin.emplace_back(0);
    bdia->dia_begin[0]=0;
    for(int i =0;i<tilesize-1;i++)
    {
        int per_tile_data_len = diaoffsetvecdis[i].size()*(rowbeginvec[i+1]-rowbeginvec[i]);
        alldatasize += per_tile_data_len;
        databegin.emplace_back(alldatasize);
        bdia->dia_begin[i+1] = bdia->dia_begin[i]+diaoffsetvecdis[i].size();
    }

    //测试原因，使用diamap.size()代替m
    alldatasize += (m-rowbeginvec[tilesize-1])*diaoffsetvecdis[tilesize-1].size();

    // cout<<"alldatasize:"<<alldatasize<<endl;
    // for(int i =0;i<tilesize;i++)
    // {
    //     cout<<bdia->dia_begin[i]<<endl;
    // }

    #ifdef DISPLAY_ROW_BEGIN
        for(int i =0;i<rowbeginvec.size();i++)
        {
            cout<<rowbeginvec[i]<<"\t";
        }
        cout<<endl;
    #endif
    bdia->offset_val = (int *)malloc(sizeof(int)*alldianums);
    bdia->data = (float*)malloc(sizeof(float)*alldatasize);
    printf("bdia零元填充:%ld\n",alldatasize-ne);
    int count = 0;
    // #pragma omp parallel for
    for(int i =0;i<tilesize;i++)
    {
        bdia->data_begin[i] = databegin[i];
        for(int j=0;j<diaoffsetvecdis[i].size();j++)
        {
            bdia->offset_val[count] = diaoffsetvecdis[i][j];
            count++;
        }
    }
    //存入data
    memset(bdia->data,0,sizeof(float)*alldatasize);
    // #pragma omp parallel for
    for(int i=0;i<ne;i++)
    {
        int tile_index = upper_bound(rowbeginvec.begin(),rowbeginvec.end(),coo_elements[i].row)-rowbeginvec.begin()-1;//计算在哪一块
        bdia->row_tile_index[coo_elements[i].row] = tile_index;
        int index = databegin[tile_index];//计算前n块的个数
        int offset = find(diaoffsetvecdis[tile_index].begin(),diaoffsetvecdis[tile_index].end(),coo_elements[i].col-coo_elements[i].row)-diaoffsetvecdis[tile_index].begin();
        // printf("coo_elements[i].row:%d,rowbeginvec[tile_index]:%d,tile_index:%d,index:%d,offset:%d\n",coo_elements[i],rowbeginvec[tile_index],tile_index,index,offset);
        index += offset*(rowbeginvec[tile_index+1]-rowbeginvec[tile_index]);//第几根对角线
        index += coo_elements[i].row-rowbeginvec[tile_index];//该对角线的第几个
        // printf("index:%d\n",index);
        *(bdia->data+index) = coo_elements[i].value;
    }



    #ifdef DISPLAY_BDIA_INFO
        display_bdia_info(bdia,alldianums,alldatasize);
    #endif
    //向gpu传递数据
    struct BDIA *bdia_d;
    int *row_begin_d,*val_size_d,*row_tile_index_d,*dia_begin_d;
    // int *for_begin_d;
    long *databegin_d;
    int *offset_val_d;
    float *data_d;
    float* vector_d;
    float* result_d;

    float *x = (float*)malloc(sizeof(float)*2*m);
    for(int i = 0;i<2*m;i++)
        x[i]=1;

    cudaMalloc((void**)&row_begin_d, sizeof(int)*(bdia->tile_size+1));
    cudaMalloc((void**)&row_tile_index_d, sizeof(int)*m);
    cudaMalloc((void**)&databegin_d, sizeof(long)*bdia->tile_size);
    cudaMalloc((void**)&dia_begin_d, sizeof(int)*bdia->tile_size);
    // cudaMalloc((void**)&for_begin_d, sizeof(int)*2*m);
    cudaMalloc((void**)&val_size_d, sizeof(int)*bdia->tile_size);
    cudaMalloc((void**)&offset_val_d, sizeof(int)*alldianums);
    cudaMalloc((void**)&data_d, sizeof(float)*alldatasize);
    cudaMemcpy(row_begin_d, bdia->row_begin, sizeof(int)*(bdia->tile_size+1), cudaMemcpyHostToDevice);
    cudaMemcpy(row_tile_index_d, bdia->row_tile_index, sizeof(int)*m, cudaMemcpyHostToDevice);
    cudaMemcpy(databegin_d, bdia->data_begin, sizeof(long)*bdia->tile_size, cudaMemcpyHostToDevice);
    cudaMemcpy(val_size_d, bdia->val_size, sizeof(int)*bdia->tile_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dia_begin_d, bdia->dia_begin, sizeof(int)*bdia->tile_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(for_begin_d, bdia->for_begin, sizeof(int)*2*m, cudaMemcpyHostToDevice);
    cudaMemcpy(offset_val_d, bdia->offset_val, sizeof(int)*alldianums, cudaMemcpyHostToDevice);
    cudaMemcpy(data_d, bdia->data, sizeof(float)*alldatasize, cudaMemcpyHostToDevice);

    bdia->row_begin=row_begin_d;
    bdia->val_size=val_size_d;
    bdia->offset_val=offset_val_d;
    bdia->data=data_d;
    bdia->data_begin=databegin_d;
    bdia->row_tile_index=row_tile_index_d;
    bdia->dia_begin = dia_begin_d;
    // bdia->for_begin = for_begin_d;
    cudaMalloc((void**)&bdia_d, sizeof(struct BDIA));
    cudaMemcpy(bdia_d, bdia, sizeof(struct BDIA), cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&vector_d, sizeof(float) * 2*m);
    cudaMemcpy(vector_d, x, sizeof(float) * 2*m, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&result_d, sizeof(float) * m);

    // 启动 CUDA 内核,唤醒一次


    int block_size = 1024;
    int num_blocks = (m - 1) / block_size + 1;
    //使用共享内存share memory
    int share_size = allmax+1024-allmin;
    int using_share = 0;
    if(share_size<=5000)
        using_share = 1;
    if(using_share)
    {
        printf("use share memory!\n");
        bdia_spmv_kernel_with_share<<<num_blocks, block_size,share_size*sizeof(float)>>>(bdia_d, vector_d, result_d, m,allmin,allmax,share_size);
    }
    else
    {
        printf("no use share memory!\n");
        bdia_spmv_kernel<<<num_blocks, block_size>>>(bdia_d, vector_d, result_d, m);
    }
    // printf("aaa:%d\n");
    
    #ifdef COMPUTE_GFLOPS
        float time_elapsed=0;
        cudaEvent_t start,stop;
        cudaEventCreate(&start);    //创建Event
        cudaEventCreate(&stop);
        if(using_share)
        {
            cudaEventRecord(start,0);
            for(int i = 0;i<timing_iterations; i++)
            {
                bdia_spmv_kernel_with_share<<<num_blocks, block_size,share_size*sizeof(float)>>>(bdia_d, vector_d, result_d, m,allmin,allmax,share_size);
            }
            cudaEventRecord( stop,0);
            cudaEventSynchronize(start);
            cudaEventSynchronize(stop);
        }
        else
        {
            cudaEventRecord(start,0);
            for(int i = 0;i<timing_iterations; i++)
            {
                bdia_spmv_kernel<<<num_blocks, block_size>>>(bdia_d, vector_d, result_d, m);
            }
            cudaEventRecord( stop,0);
            cudaEventSynchronize(start);
            cudaEventSynchronize(stop);
        }
        cudaEventElapsedTime(&time_elapsed,start,stop);
        time_elapsed /= 1000.0 * (float) timing_iterations;

        double gflops = 2*(ne);
        gflops /=time_elapsed * 1000000000;
        printf("bdia gflops:%f\n",gflops);

    // //该处lightspmv是2*（m+ne），而merge是2*ne
    #endif


    // 从 GPU 获取结果,打印结果
    #ifdef CHECK_RESULT
        float *result_h = (float*)malloc(sizeof(float) * m);
        cudaMemcpy(result_h, result_d, sizeof(float) * m, cudaMemcpyDeviceToHost);
        checkResult(result_cpu,result_h,m);
    #endif
    // for(int i =0;i<m;i++)
    // {
    //     printf("%f\n",result_h[i]);
    // }

}

int main(int argc, char **argv)
{
    int symmetric;
    int m,n,ne,timing_iterations;
    MM_typecode matcode;
    FILE *f;
    timing_iterations = 1000;
    int nrows = 512;
    if ((f = fopen(argv[1], "r")) == NULL)
        exit(1);
    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }
    symmetric = mm_is_symmetric(matcode);
    // printf("symmetric:%d\n",symmetric);
    mm_read_mtx_crd_size(f, &m, &n, &ne);
    int row_tmp,col_tmp;
    float val_tmp;
    struct COOElement *coo_elements;
    if(symmetric)
    {
        int count = 0;
        coo_elements = (struct COOElement*)malloc(sizeof(struct COOElement)*ne*2);
        // #pragma omp parallel for//可能有问题
        for (int i = 0; i < ne; i++)
        {
            fscanf(f, "%d %d %f\n", &row_tmp, &col_tmp, &val_tmp);
            row_tmp --;
            col_tmp --;
            coo_elements[count++] = {row_tmp, col_tmp, val_tmp};
            if(row_tmp != col_tmp)
                coo_elements[count++] = {col_tmp, row_tmp, val_tmp};
        }
        ne = count;
    }
    else
    {
        coo_elements = (struct COOElement*)malloc(sizeof(struct COOElement)*ne);
        // #pragma omp parallel for//可能有问题
        for (int i = 0; i < ne; i++)
        {
            fscanf(f, "%d %d %f\n", &row_tmp, &col_tmp, &val_tmp);
            row_tmp --;
            col_tmp --;
            coo_elements[i] = {row_tmp, col_tmp, val_tmp};
        }
    }
    // printf("reading file finish\n");
    // printf("coo to dia");
    struct DIAFormat *dia_matrix;
    dia_matrix = (struct DIAFormat*)malloc(sizeof(struct DIAFormat)*ne);
    convertCOOtoDIA(coo_elements, ne, m, n, dia_matrix);
    // 打印DIA格式的数据
    displayInfo(m, n, ne,dia_matrix);
    float *x = (float*)malloc(sizeof(float)*m);
    float *result_cpu = (float*)malloc(sizeof(float)*m);
    // #pragma omp parallel for
    for(int i = 0; i < m; i++)
    {
        x[i] = 1;
    }
    // CPU运行速度有点慢，后续考虑openmp优化
    memset(result_cpu,0,sizeof(float)*m);
    #ifdef CHECK_RESULT
        spmv_cpu(dia_matrix, x, result_cpu, m);
    #endif
    // printf("cpu Result after SPMV:\n");
    // for (int i = 0; i < m; ++i) {
    //     printf("%f\n", result_cpu[i]);
    // }
    // printf("\n");

    // 分配和传输数据到 GPU
    int dev = 0;
    int maxdiff = 0;
    if(argc==3)
        nrows = atoi(argv[2]);
    if(argc==4)
    {
        nrows = atoi(argv[2]);
        maxdiff = atoi(argv[3]);
    }
    if(argc==5)
    {
        nrows = atoi(argv[2]);
        maxdiff = atoi(argv[3]);
        dev = atoi(argv[4]);
        
    }
    // printf("Use device:%d\n",dev);
    printf("nrows: %d\t",nrows);
    printf("maxdiff: %d\t",maxdiff);
    printf("行列数: %d\t",m);
    printf("对角线数: %d\t",dia_matrix->num_diagonals);
    printf("原始非零元个数:%d\n",ne);
    cudaSetDevice(dev);
    gpu_test_dia(dia_matrix,x,m,ne,timing_iterations,result_cpu);  
    gpu_test_brcsd2(dia_matrix,x,m,ne,timing_iterations,result_cpu,nrows);

    // gpu_test_brcsd2_2(dia_matrix,x,m,ne,timing_iterations,result_cpu,nrows);

    gpu_test_bdia(coo_elements,m,ne,timing_iterations,result_cpu,maxdiff);
    return 0;
}