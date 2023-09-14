#include<cuda_runtime.h>
#include<stdio.h>
#include"util.h"
int main(int argc,char **argv)
{
    //设置gpu设备
    int dev = 2;
    cudaSetDevice(dev);

    /*
    Kernel核函数编写有以下限制
    只能访问设备内存
    必须有void返回类型
    不支持可变数量的参数
    不支持静态变量
    显示异步行为
    */
    cudaError_t cudaDeviceSynchronize(void);//主机等待设备端执行
    cudaError_t cudaMemcpy(void* dst,const void * src,size_t count,cudaMemcpyKind kind);
    //隐式方法，隐式方法就是不明确说明主机要等待设备端，而是设备端不执行完，主机没办法进行,
    /*
    每个SM上有多个block，一个block有多个线程（可以是几百个，但不会超过某个最大值），但是从机器的角度
    ，在某时刻T，SM上只执行一个线程束，也就是32个线程在同时同步执行
    */

    /*
    3.2
    因为线程束分化导致的性能下降就应该用线程束的方法解决，根本思路是避免同一个线程束内的线程分化，
    而让我们能控制线程束内线程行为的原因是线程块中线程分配到线程束是有规律的而不是随机的。
    这就使得我们根据线程编号来设计分支是可以的，补充说明下，当一个线程束中所有的线程都执行if或者，都执行else时，不存在性能下降；
    只有当线程束内有分歧产生分支的时候，性能才会急剧下降。
    */
}