#include <cuda.h>
#include <cuda_runtime.h>

#define N (33 * 1024)

__global__ void add(int *a, int *b, int *c)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while(tid < N)
    {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

void cudaAdd(int *a, int *b, int *c)
{
    add<<<128, 128>>>(a, b, c);
}
