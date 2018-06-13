#include <QCoreApplication>
#include <stdio.h>
#include <cuda_runtime.h>

#define N 10

void cudaAdd(int *a, int *b, int *c);

int main(void)
{
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // 在gpu上分配内存
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    // 在cpu上为数组'a'和'b'赋值
    for(int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i * i;
    }

    // 将数组'a'和'b'复制到gpu
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaAdd(dev_a, dev_b, dev_c);

    // 将数组'c'从gpu复制到cpu
    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // 显示结果
    for(int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
