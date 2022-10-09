#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n)
{
  // set thread ID
  unsigned int tid = threadIdx.x;
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  
  // convert global data pointer to the local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x;

  // boundary check
  if (idx >= n) return;

  // in-place reduction in global memory
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if ((tid % (2*stride)) == 0) {
      idata[tid] += idata[tid + stride];
    }
    
    // synchronize within block
    __syncthreads();
  }

  // write result for this block to global memory
  if (tid == 0)
    g_odata[blockIdx.x] = idata[0];
}

int recursiveReduce(int *data, const int size)
{
  // terminate check
  if (size == 1) return data[0];

  // renew the stride
  int stride = size / 2;

  // in-place reduction
  for (int i = 0; i < stride; i++) {
    data[i] += data[i+stride];
  }

  // call recursively
  return recursiveReduce(data, stride);
}

__global__ void gpuRecursiveReduce(int *g_idata, int *g_odata, unsigned int isize)
{
  // set thread ID
  unsigned int tid = threadIdx.x;

  // convert global data pointer to the local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x;
  int *odata = &g_odata[blockIdx.x];

  // stop condition
  if (isize == 2 && tid == 0) {
    g_odata[blockIdx.x] = idata[0] + idata[1];
    return;
  }

  // nested invocation
  int istride = isize >> 1;
  if (istride > 1 && tid < istride) {
    // inplace reduction
    idata[tid] += idata[tid + istride];
  }

  // sync at block level
  __syncthreads();

  // nested invocation
  if (tid == 0) {
    gpuRecursiveReduce<<<1, istride>>>(idata, odata, istride);

    // sync all child grids launched in this block
    cudaDeviceSynchronize();
  }

  // sync at block level again
  __syncthreads();
}

__global__ void gpuRecursiveReduceNosync(int *g_idata, int *g_odata, unsigned int isize)
{
  // set thread ID
  unsigned int tid = threadIdx.x;

  // convert global data pointer to the local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x;
  int *odata = &g_odata[blockIdx.x];

  // stop condition
  if (isize == 2 && tid == 0) {
    g_odata[blockIdx.x] = idata[0] + idata[1];
    return ;
  }

  // nested invocation
  int istride = isize >> 1;
  if (istride > 1 && tid < istride) {
    // inplace reduction
    idata[tid] += idata[tid + istride];
  }

  // nested invocation
  if (tid == 0) {
    gpuRecursiveReduceNosync<<<1, istride>>>(idata, odata, istride);
  }
}

__global__ void gpuRecursiveReduce2(int *g_idata, int *g_odata, unsigned int iStride, const int iDim)
{
  // convert global data pointer to the local pointer of this block
  int *idata = g_idata + blockIdx.x * iDim;

  // stop condition
  if (iStride == 1 && threadIdx.x == 0) {
    g_odata[blockIdx.x] = idata[0] + idata[1];
    return ;
  }

  // nested invocation
  idata[threadIdx.x] += idata[threadIdx.x + iStride];

  // nested invocation
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    gpuRecursiveReduce2<<<1, iStride/2>>>(g_idata, g_odata, iStride/2, iDim);
  }
}

double seconds()
{
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (double)tp.tv_sec*1e3 + (double)tp.tv_usec*1.e-3;
}

int main(int argc, char *argv[])
{
  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("%s starting reduction at ", argv[0]);
  printf("device %d: %s ", dev, deviceProp.name);
  cudaSetDevice(dev);

  bool bResult = false;

  // initialization
  int size = 1 << 24;  // total number of elements to reduce
  printf("    with array size %d ", size);

  // execution configuration
  int blocksize = 512;  // initial block size
  if (argc > 1) {
    blocksize = atoi(argv[1]);  // block size form command line argument
  }
  dim3 block(blocksize, 1);
  dim3 grid((size+block.x-1)/block.x, 1);
  printf("grid %d block %d\n", grid.x, block.x);

  // allocate host memory
  size_t bytes = size * sizeof(int);
  int *h_idata = (int *)malloc(bytes);
  int *h_odata = (int *)malloc(grid.x*sizeof(int));
  int *tmp = (int *)malloc(bytes);

  // initialize the array
  for (int i = 0; i < size; i++) {
    // mask off high 2 bytes to force max number to 255
    h_idata[i] = (int)(rand() & 0xFF);
  }
  memcpy(tmp, h_idata, bytes);

  double iStart, iElaps;
  int gpu_sum = 0;

  // allocate device memory
  int *d_idata = NULL;
  int *d_odata = NULL;
  cudaMalloc((void **)&d_idata, bytes);
  cudaMalloc((void **)&d_odata, grid.x*sizeof(int));

  //cpu reduction
  iStart = seconds();
  int cpu_sum = recursiveReduce(tmp, size);
  iElaps = seconds() - iStart;
  printf("cpu reduce                          elapsed %8d ms cpu_sum: %d\n", iElaps, cpu_sum);

  // kernel warmup: reduceNeighbored
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  iStart = seconds();
  reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
  cudaDeviceSynchronize();
  iElaps = seconds() - iStart;
  cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++)
    gpu_sum += h_odata[i];
  printf("gpu Warmup                          elapsed %8.4lf ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

  // kernel 1: reduceNeighbored
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  iStart = seconds();
  reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
  cudaDeviceSynchronize();
  iElaps = seconds() - iStart;
  cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++)
    gpu_sum += h_odata[i];
  printf("gpu Neighbored                      elapsed %8.4lf ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

  // kernel 2: GPURecursiveReduce
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  iStart = seconds();
  gpuRecursiveReduce<<<grid, block>>>(d_idata, d_odata, size);
  cudaDeviceSynchronize();
  iElaps = seconds() - iStart;
  cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++)
    gpu_sum += h_odata[i];
  printf("gpu gpuRecursiveReduce              elapsed %8.4lf ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

  // kernel 3: GPURecursiveReduceNosync
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  iStart = seconds();
  gpuRecursiveReduceNosync<<<grid, block>>>(d_idata, d_odata, size);
  cudaDeviceSynchronize();
  iElaps = seconds() - iStart;
  cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++)
    gpu_sum += h_odata[i];
  printf("gpu GPURecursiveReduceNosync        elapsed %8.4lf ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

  // kernel 4: GPURecursiveReduce2
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  iStart = seconds();
  gpuRecursiveReduce2<<<grid, block>>>(d_idata, d_odata, block.x/2, block.x);
  cudaDeviceSynchronize();
  iElaps = seconds() - iStart;
  cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++)
    gpu_sum += h_odata[i];
  printf("gpu GPURecursiveReduce2             elapsed %8.4lf ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);
 

  // free host memory
  free(h_idata);
  free(h_odata);
  free(tmp);

  // free device memory
  cudaFree(d_idata);
  cudaFree(d_odata);

  // check the  results
  bResult = (gpu_sum == cpu_sum);
  if (!bResult)
    printf("Test failed!\n");
  
  return 0;
}