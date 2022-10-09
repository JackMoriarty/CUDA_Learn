#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void nestedHelloWorld(const int iSize, int iDepth)
{  
  int tid = threadIdx.x;
  printf("Recursion=%d: Hello World from thread %d "
	 "block %d\n", iDepth, tid, blockIdx.x);

  // condition to stop recursive execution
  if (iSize == 1)
    return;

  // reduce block size to half
  int nthreads = iSize >> 1;

  // thread 0 launches child grid recursively
  if (tid == 0 && nthreads > 0) {
    nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth);
    printf("--------> nested execution depth: %d\n", iDepth);
  }
}

int main(int argc, char *argv[])
{
  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("%s starting at ", argv[0]);
  printf("device %d: %s\n", dev, deviceProp.name);

  // execution configuration
  int gridsize = 1;
  int blocksize = 32;
  if (argc > 1)
    gridsize = atoi(argv[1]);

  printf("%s Execution Configuration: grid %d block %d\n", argv[0], gridsize, blocksize);

  nestedHelloWorld<<<gridsize, blocksize>>>(blocksize, 0);
  cudaDeviceSynchronize();

  return 0;
}
