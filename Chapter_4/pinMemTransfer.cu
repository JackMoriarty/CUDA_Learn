#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
  // set up device
  int dev = 0;
  cudaSetDevice(dev);

  // memory size
  unsigned int isize = 1 << 22;
  unsigned int nbytes = isize * sizeof(float);

  // get device information
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("%s starting at ", argv[0]);
  printf("device %d: %s memory size %d nbyte %5.2fMB\n", dev, deviceProp.name,
	 isize, nbytes/(1024.0f*1024.0f));

  // allocate the pin host memory
  float *h_a_pin = NULL;
  cudaError_t status = cudaMallocHost((void **)&h_a_pin, nbytes);
  if (status != cudaSuccess) {
    fprintf(stderr, "Error returned from pinned host memory allocation\n");
    exit(1);
  }

  // allocate the device memory
  float *d_a;
  cudaMalloc((float **)&d_a, nbytes);

  // initialize the host memory
  for (int i = 0; i < isize; i++) h_a_pin[i] = 0.5f;

  // transfer data from the host to the device
  cudaMemcpy(d_a, h_a_pin, nbytes, cudaMemcpyHostToDevice);

  // transfer data from the device to the host
  cudaMemcpy(h_a_pin, d_a, nbytes, cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_a);
  cudaFreeHost(h_a_pin);

  //reset device
  cudaDeviceReset();
  return EXIT_SUCCESS;
}