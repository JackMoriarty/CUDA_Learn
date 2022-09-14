#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#define CHECK(call)							\
{									\
  const cudaError_t error = call;					\
  if (error != cudaSuccess) {						\
    printf("Error: %s:%d, ", __FILE__, __LINE__);			\
    printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));	\
    exit(1);								\
  }									\
}

double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (double)tp.tv_sec + (double)tp.tv_usec*1.e-6;
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
  float *ia = A;
  float *ib = B;
  float *ic = C;

  for (int iy = 0; iy < ny; iy++) {
    for (int ix = 0; ix < nx; ix++) {
      ic[ix] = ia[ix] + ib[ix];
    }
    ia += nx;
    ib += nx;
    ic += nx;
  }
}

void initialData(float *ip, int size)
{
  // generate different seed for random numbers
  time_t t;
  srand((unsigned int)time(&t));

  for (int i = 0; i < size; i++) {
    ip[i] = (float)(rand() & 0xFF) / 10.0f;            
  }
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
  double epsilon = 1.0E-8;
  bool match = true;
  for (int i = 0; i < N; i++) {
    if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
      match = false;
      printf("Array do not match!\n");
      printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
      break;
    }
  }

  if (match) printf("Array match.\n");
}

__global__ void sumMatrixOnGPU1D(float * MatA, float *MatB, float *MatC,
				 int nx, int ny)
{
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;

  if (ix < nx) {
    for (int iy = 0; iy < ny; iy++) {
      int idx = iy*nx + ix;
      MatC[idx] = MatA[idx] + MatB[idx];
    }
  }

}


int main(int argc, char *argv[])
{
  printf("%s Starting...\n", argv[0]);

  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("Using Device %d: %s\n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));

  // set up data size of matrix
  int nx = 1 << 12;
  int ny = 1 << 12;

  int nxy = nx*ny;
  int nBytes = nxy*sizeof(float);
  
  // malloc host memroy
  float *h_A, *h_B, *hostRef, *gpuRef;
  h_A = (float *)malloc(nBytes);
  h_B = (float *)malloc(nBytes);
  hostRef = (float *)malloc(nBytes);
  gpuRef = (float *)malloc(nBytes);

  // initialize data at host side
  double iStart = cpuSecond();
  initialData(h_A, nxy);
  initialData(h_B, nxy);
  double iElaps = cpuSecond() - iStart;

  memset(hostRef, 0, nBytes);
  memset(gpuRef, 0, nBytes);

  // add matrix at host side for result checks
  iStart = cpuSecond();
  sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
  iElaps = cpuSecond() - iStart;

  // malloc device global memory
  float *d_MatA, *d_MatB, *d_MatC;
  cudaMalloc(&d_MatA, nBytes);
  cudaMalloc(&d_MatB, nBytes);
  cudaMalloc(&d_MatC, nBytes);

  // transfer data from host to device
  cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

  // invoke kernel at host side
  dim3 block(32, 1);
  dim3 grid((nx+block.x-1)/block.x, 1);

  iStart = cpuSecond();
  sumMatrixOnGPU1D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
  cudaDeviceSynchronize();
  iElaps = cpuSecond() - iStart;
  printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", \
	 grid.x, grid.y, block.x, block.y, iElaps);

  // copy kernel results
  cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

  // check device results
  checkResult(hostRef, gpuRef, nxy);

  // free device global memory results
  cudaFree(d_MatA);
  cudaFree(d_MatB);
  cudaFree(d_MatC);

  // free host memory
  free(h_A);
  free(h_B);
  free(hostRef);
  free(gpuRef);

  // reset device
  cudaDeviceReset();
  
  return 0;
}