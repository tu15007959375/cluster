#include <stdlib.h>
#include <stdio.h>
#include "sys/time.h"
#include "struct.h"
#define MIN(i, j) (((i) < (j)) ? (i) : (j))
#define MAX(i, j) (((i) > (j)) ? (i) : (j))
int cmpfunc(const void * a, const void * b)
{
   return ( *(int*)a - *(int*)b );
}
void displayInfo(int m, int n, int ne, struct DIAFormat *dia_matrix, bool displayInfo_enable = false)
{
    if(!displayInfo_enable)
        return;
    printf("%d %d %d\n", m, n, ne);
    printf("DIA格式:\n");
    printf("对角线数量: %d\n", dia_matrix->num_diagonals);
    printf("最大对角线长度: %d\n", dia_matrix->max_diag_length);
    
    printf("对角线偏移数组:\n");
    for (int i = 0; i < dia_matrix->num_diagonals; ++i) {
        printf("%d ", dia_matrix->diagonals_offsets[i]);
    }
    printf("\n");

    printf("对角线值数组:\n");
    for (int i = 0; i < dia_matrix->num_diagonals; ++i) {
        for (int j = 0; j < dia_matrix->max_diag_length; ++j) {
            printf("%f ", dia_matrix->diagonals_values[i * dia_matrix->max_diag_length + j]);
        }
        printf("\n");
    }
}
void checkResult(float* result,float* result_d,int m)
{
    for(int i = 0; i<m; i++)
    {
        if(result[i]!=result_d[i])
        {
            printf("CHECK FAIL!\n");
            return;
        }
    }
    printf("CHECK PASS!\n");
}
void del_index_num(int *p, int len, int index)
{
    if(index >= len)
        return;
    for(int i = index+1;i<len;i++)
        p[i-1] = p[i];
}
void del_index_num2(int *p, int len, int start,int nums)
{
    if(start+nums>=len)
        return;

    for(int i = start+nums;i<len;i++)
        p[i-nums] = p[i];
}

void display_brcsd2_info(const struct BRCSD2* brcsd2)
{
    printf("总块数:%d\n",brcsd2->tile_size);
    printf("实际块数:%d\n",brcsd2->ac_size);
    int start=0;
    for(int i = 0;i<brcsd2->ac_size;i++)
    {
        printf("第%d块包含数量:%d 对角线数量:%d\n对角线值:",i,brcsd2->offset_key[i],brcsd2->val_size[i]);
        for(int j =0;j<brcsd2->val_size[i];j++)
        {   
            printf("%d\t",*(brcsd2->offset_val+start+j));
        }
        start +=brcsd2->val_size[i];
        printf("\n");
    }
    start=0;
    for(int i =0;i<brcsd2->tile_size;i++)
        start += brcsd2->val_size[i];
    for(int i =0;i<brcsd2->nrows*start;i++)
        printf("%f\t",brcsd2->data[i]);
    // for(int i =0;i<start*nrows;i++)
    //     printf("%f\t",brcsd2->data[i]);
    // start=0;
    // for(int i = 0;i<brcsd2->ac_size;i++)
    // {
    //     //TODO 该处在块大小大于行数，或者最后一块时，输出有问题
    //     for(int j =0;j<brcsd2->val_size[i]*brcsd2->offset_key[i]*brcsd2->nrows;j++)
    //     {
    //         printf("%f\t",*(brcsd2->data+j+start));
    //     }
    //     start += brcsd2->val_size[i]*brcsd2->offset_key[i]*brcsd2->nrows;
    //     printf("\n");
    // }
}
bool is_all_zero(float *p,int start,int end)
{
    for(int i =start;i<end;i++)
    {
        if(p[i]!=0)
            return false;
    }
    return true;
}
void display_bdia_info(const struct BDIA* bdia,int alldianums,long alldatasize)
{
    printf("总块数:%d\n",bdia->tile_size);
    printf("总对角线数:%d\n",alldianums);
    printf("总元素个数:%ld\n",alldatasize);
    printf("每块行偏移:");
    for(int i =0;i<bdia->tile_size;i++)
    {
        printf("%d ",bdia->row_begin[i]);
    }
    printf("\n每块对角线数量:");
    for(int i =0;i<bdia->tile_size;i++)
    {
        printf("%d->%d ",i,bdia->val_size[i]);
    }
    printf("\n每块对角线偏移:\n");
    int count = 0;
    for(int i =0;i<bdia->tile_size;i++)
    {
        printf("%d->",i);
        for(int j =0;j<bdia->val_size[i];j++)
        {
            printf("%d ",bdia->offset_val[count]);
            count++;
        }
        printf("\n");
    }
    printf("值:\n");
    for(int i =0;i<alldatasize;i++)
        printf("%f\t",bdia->data[i]);
    printf("\n");
}
// struct BRCSD1 {
//     int block_numbers;//块的数量,ex:2
//     int *offset_key;//每块的起始位置,ex:0,2,6
//     int *val_size;//每块的dia对角线长度,,ex:3,3
//     int **offset_val;//每块的dia偏移,二维数组,ex:[0,1,4],[-2,0,1]
//     float **data;//数据,二维数组,ex:[1,4,2,5,3,6],[7,10,13,16,8,11,14,17,9,12,15,0]
// };
// int generate_p(struct DIAFormat* dia_matrix,int *p, int m,int eta)
// {
//     int len = dia_matrix->num_diagonals+1;
//     //cal p
//     for(int i =0; i<len-1; i++)
//     {
//         if(dia_matrix->diagonals_offsets[i]>0)
//         {
//             p[i] = m-1-dia_matrix->diagonals_offsets[i];
//         }
//         else
//         {
//             p[i] = -dia_matrix->diagonals_offsets[i];
//         }
//     }
//     p[len-1] = m-1;
//     qsort(p, len-1, sizeof(int), cmpfunc);

//     printf("eta:%d\np初始化:\n",eta);
//     for(int i=0;i<len;i++)
//         printf("%d\t",p[i]);

//     int left = 0;
//     int right = 1;
//     while(right<len)
//     {
//         if(p[right]-p[left]+1 >= eta)
//             break;
//         else
//         {
//             if(right == len-1)
//             {
//                 left ++;
//                 right = left+1;
//             }
//             else
//                 right++;
//         }
//     }
//     if(right >= len)
//     {
//         printf("cal distance error!!!,left:%d,right:%d,len:%d\n",left,right,len);
//         return -1;
//     }
//     if(right-left > 1)
//     {
//         while(right < len)
//         {
//             p[right-1] = p[right];
//             right++;
//         }
//         len = right-1;
//     }
//     //至此为止完成了步骤一
//     left ++;
//     right = left+1;
//     while(right < len)
//     {
//         if(p[right]-p[left]<eta)
//         {
//             if(right == len-1)
//                 del_index_num(p,len,right-1);
//             else
//                 del_index_num(p,len,right);
//             len--;
//         }
//         else
//         {
//             left++;
//             right++;
//         }
//     }
//     printf("\n处理后:\n");
//     for(int i =0;i<len;i++)
//     {
//         printf("p:%d\t",p[i]);
//     }
//     return len;
// }
// void gpu_test_brcsd1(struct DIAFormat* dia_matrix,float *x,int m,int ne,int timing_iterations)
// {
//     int eta = 2;
//     int *p = (int *)malloc(sizeof(int)*(dia_matrix->num_diagonals+1));
//     int len = generate_p(dia_matrix,p,m,eta)-1;
//     if(len <= 0)
//         return;
//     //生成brcsd1
//     // len = 4;
//     // p[0]=0;
//     // p[1]=2;
//     // p[2]=5;
//     // p[3]=7;
//     // p[4]=9;
//     struct BRCSD1 *brcsd1;
//     brcsd1 = (struct BRCSD1*)malloc(sizeof(struct BRCSD1));
//     brcsd1->block_numbers = len;
//     brcsd1->offset_key = (int*)malloc(sizeof(int)*(len)+1);
//     brcsd1->val_size = (int*)malloc(sizeof(int)*(len));
//     brcsd1->offset_key[0] = 0;
//     for(int i = 1;i <=len; i++)
//     {
//         brcsd1->offset_key[i] = p[i]+1;
//     }
//     // for(int i = 0;i <=len; i++)
//     //     printf("brcsd1->offset_key[i]:%d\t", brcsd1->offset_key[i]);
//     brcsd1->offset_val = (int **)malloc(sizeof(int*)*len);
//     brcsd1->data = (float **)malloc(sizeof(float*)*len);
//     int max_diag_length = dia_matrix->max_diag_length;
//     //计算val_size
//     for(int i = 0;i<len;i++)
//     {
//         int count = 0;
//         for(int j =0;j<dia_matrix->num_diagonals;j++)
//         {
//             bool flag = is_all_zero(dia_matrix->diagonals_values+max_diag_length*j,brcsd1->offset_key[i],brcsd1->offset_key[i+1]);
//             if(!flag)
//             {
//                 count++;
//             }
//             // if(flag)
//             //     printf("dia_matrix->num_diagonals:%d,brcsd1->offset_key[i]:%dbrcsd1->offset_key[i+1]:%d\n",j,brcsd1->offset_key[i],brcsd1->offset_key[i+1]);
//         }
//         brcsd1->val_size[i]=count;
//     }
//     //分配offset_val和data
//     for(int i = 0;i<len;i++)
//     {
//         brcsd1->offset_val[i] = (int *)malloc(sizeof(int )*brcsd1->val_size[i]);
//         brcsd1->data[i] = (float *)malloc(sizeof(float )*(brcsd1->offset_key[i+1]-brcsd1->offset_key[i])*brcsd1->val_size[i]);
//     }
//     for(int i = 0;i<len;i++)
//     {
//         int count = 0;
//         int count2 = 0;
//         int size = brcsd1->offset_key[i+1] - brcsd1->offset_key[i];
//         for(int j =0;j<dia_matrix->num_diagonals;j++)
//         {
//             bool flag = is_all_zero(dia_matrix->diagonals_values+max_diag_length*j,brcsd1->offset_key[i],brcsd1->offset_key[i+1]);
//             if(!flag)
//             {
//                 brcsd1->offset_val[i][count] = dia_matrix->diagonals_offsets[j];
//                 for(int k = 0;k < size; k++)
//                 {
//                     brcsd1->data[i][count2] = *(dia_matrix->diagonals_values+max_diag_length*j+k+brcsd1->offset_key[i]);
//                     count2++;
//                 }
//                 // printf("dia_matrix->diagonals_offsets:%d\t",dia_matrix->diagonals_offsets[j]);
//                 count++;
//             }
//         }
//         // printf("size:%d",size);
//     }
//     // for(int i = 0;i<len;i++)
//     //     for(int j =0;j<(brcsd1->offset_key[i+1]-brcsd1->offset_key[i])*brcsd1->val_size[i];j++)
//     //         printf("\n块数:%d,值:%f",i,brcsd1->data[i][j]);
    
// }