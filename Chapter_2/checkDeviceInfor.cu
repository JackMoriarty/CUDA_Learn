#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
  printf("%s Starting...\n", argv[0]);

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n",
	   (int)error_id, cudaGetErrorString(error_id));
    printf("Result = FALL\n");
    exit(EXIT_FAILURE);
  }

  if (deviceCount == 0) {
    printf("There are no available device(s) that support CUDA\n");
  } else {
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  }

  int dev = 0, driverVersion = 0, runtimeVersion = 0;
  cudaSetDevice(dev);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("Device %d: \"%s\"\n", dev, deviceProp.name);

  cudaRuntimeGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  printf("\u251c\u2500\u2500 CUDA Driver Version / Runtime Version:          %d.%d / %d.%d\n",
	 driverVersion/1000, (driverVersion%100)/10,
	 runtimeVersion/1000, (runtimeVersion%100)/10);
  printf("\u251c\u2500\u2500 CUDA Capability Major/Minor version number:     %d.%d\n",
	 deviceProp.major, deviceProp.minor);
  printf("\u251c\u2500\u2500 Total amount of global memory:                  %.2f MBytes (%llu bytes)\n",
	 (float)deviceProp.totalGlobalMem/(pow(1024, 0.3)),
	 (unsigned long)deviceProp.totalGlobalMem);
  printf("\u251c\u2500\u2500 GPU Clock rate:                                 %.0f MHz (%0.2f GHz)\n",
	 deviceProp.clockRate*1e-3f, deviceProp.clockRate*1e-6f);
  printf("\u251c\u2500\u2500 Memory Clock rate:                              %.0f MHz\n",
	 deviceProp.memoryClockRate*1e-3f);
  printf("\u251c\u2500\u2500 Memory Bus Width:                               %d-bit\n",
	 deviceProp.memoryBusWidth);
  if (deviceProp.l2CacheSize) {
    printf("\u251c\u2500\u2500 L2 Cache Size:                                  %d bytes\n",
	   deviceProp.l2CacheSize);
  }
  printf("\u251c\u2500\u2500 Max Texture Dimension Size (x,y,z)              "
	 "1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
	 deviceProp.maxTexture1D,
	 deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
	 deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],
	 deviceProp.maxTexture3D[2]);
  printf("\u251c\u2500\u2500 Max Layered Texture Size (dim) x layers          "
	 "1D=(%d) x %d, 2D=(%d,%d) x %d\n",
	 deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
	 deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
	 deviceProp.maxTexture2DLayered[2]);
  printf("\u251c\u2500\u2500 Total amount of const memory:                   %lu bytes\n", deviceProp.totalConstMem);
  printf("\u251c\u2500\u2500 Total amount of shared memory per block:        %lu bytes\n",
	 deviceProp.sharedMemPerBlock);
  printf("\u251c\u2500\u2500 Total number of registers available per block:  %d\n",
	 deviceProp.regsPerBlock);
  printf("\u251c\u2500\u2500 Warp size:                                      %d\n", deviceProp.warpSize);
  printf("\u251c\u2500\u2500 Maximum number of threads per multiprocessor:   %d\n",
	 deviceProp.maxThreadsPerMultiProcessor);
  printf("\u251c\u2500\u2500 Maximum number of threads per block:            %d\n",
	 deviceProp.maxThreadsPerBlock);
  printf("\u251c\u2500\u2500 Maximum sizes of each dimension of a block:     %d x %d x %d\n",
	 deviceProp.maxThreadsDim[0],
	 deviceProp.maxThreadsDim[1],
	 deviceProp.maxThreadsDim[2]);
  printf("\u251c\u2500\u2500 Maximum sizes of each dimension of a grid:      %d x %d x %d\n",
	 deviceProp.maxGridSize[0],
	 deviceProp.maxGridSize[1],
	 deviceProp.maxGridSize[2]);
  printf("\u2514\u2500\u2500 Maximum memory pitch:                           %lu bytes\n", deviceProp.memPitch);
  
  exit(EXIT_SUCCESS);
}