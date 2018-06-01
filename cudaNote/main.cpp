#include <QCoreApplication>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

const int N = 10;

void cudaAdd(int*, int*, int*);

int main(int argc, char *argv[])
{
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    cudaMalloc(&dev_a, N * sizeof(int));
    cudaMalloc(&dev_b, N * sizeof(int));
    cudaMalloc(&dev_c, N * sizeof(int));

    for(int i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }

    cudaMemcpy(dev_a, a, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeof(int) * N, cudaMemcpyHostToDevice);

    cudaAdd(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i++)
    {
        std::cout<<c[i]<<" ";
    }
    std::cout<<std::endl;

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
