#include <QCoreApplication>
#include <stdio.h>
#include <cuda_runtime.h>

#define imin(a, b) (a < b ? a : b)
const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

void dot(float *a, float *b, float *c);

int main(int argc, char *argv[])
{
//    QCoreApplication a(argc, argv);

//    return a.exec();
    float *a, *b, *partial_c, c;
    float *dev_a, *dev_b, *dev_partial_c;

    // CPU上分配内存
    a = new float[N];
    b = new float[N];
    partial_c = new float[blocksPerGrid];

    // GPU上分配内存
    cudaMalloc((void**)&dev_a, N * sizeof(float));
    cudaMalloc((void**)&dev_b, N * sizeof(float));
    cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float));

    for(int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i * 2;
    }

    cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    dot(dev_a, dev_b, dev_partial_c);

    cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    c = 0;
    for(int i = 0; i < blocksPerGrid; i++)
    {
        c += partial_c[i];
    }

    printf("%.6g\n", c);

    delete []a;
    delete []b;
    delete []partial_c;
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

}
