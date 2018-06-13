#include <cuda.h>
#include <cuda_runtime.h>

#define N 10

__global__ void add(int *a, int *b, int *c)
{
    int tid = threadIdx.x;
    if(tid < N)
        c[tid] = a[tid] + b[tid];
}

void cudaAdd(int *a, int *b, int *c)
{
    add<<<1, N>>>(a, b, c);
}
