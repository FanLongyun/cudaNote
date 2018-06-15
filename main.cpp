#include <QCoreApplication>
#include <cuda_runtime.h>
#include <stdio.h>

#define N (33 * 1024)

void cudaAdd(int *a, int *b, int *c);

int main(int argc, char *argv[])
{
//    QCoreApplication a(argc, argv);

//    return a.exec();

    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // GPU上分配内存
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    // CPU上为数组赋值
    for(int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i * i;
    }

    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaAdd(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    bool success = true;
    for(int i = 0; i < N; i++)
    {
        if(a[i] + b[i] != c[i])
        {
            success = false;
            break;
        }
    }
    if(success)
        printf("no error.\n");

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
