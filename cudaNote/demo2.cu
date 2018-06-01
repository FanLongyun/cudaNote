#include <cuda.h>
#include <cuda_runtime.h>

__global__ void add(int *a, int *b, int *c)
{
    int tid = blockIdx.x;
    if(tid < 10)
    {
        c[tid] = a[tid] + b[tid];
    }
}

void cudaAdd(int *a, int *b, int *c)
{
    add<<<10, 1>>>(a, b, c);
}
