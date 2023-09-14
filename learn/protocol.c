#include <stdio.h>
#include <stdlib.h>
int recursiveReduce(int *data, int const size)
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
    return recursiveReduce(data,stride);
}
int main()
{
    int *data;
    int size = 10;
    data = (int *)malloc(sizeof(int)*size);
    for(int i=0;i<size;i++)
        data[i]=i+1;
    recursiveReduce(data,size);
    printf("the res is : %d",data[0]);
}