#include <QtCore/QCoreApplication>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

extern "C"
void runCudaPart();

int main(int argc, char *argv[])
{
    int count;
    cudaGetDeviceCount(&count);
    cudaDeviceProp cudaProp;
    cudaGetDeviceProperties(&cudaProp, count);
    std::cout<<"device num: "<<count<<std::endl;
    runCudaPart();
}
