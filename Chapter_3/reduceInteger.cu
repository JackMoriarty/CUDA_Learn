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

__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n)
{
  // set thread ID
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

  // convert global data pointer to the local pointer of this block
  int *idata = g_idata + blockIdx.x*blockDim.x;

  // boundary check
  if (idx >= n)
    return;

  // in-place reduction in global memory
  for (int stride = 1; stride < blockDim.x; stride*= 2) {
    // convert tid into local array index
    int index = 2*stride*tid;
    if (index < blockDim.x) {
      idata[index] += idata[index+stride];
    }

    // synchronize within threadblock
    __syncthreads();
  }

  // write result for this block to global memory
  if (tid == 0)
    g_odata[blockIdx.x] = idata[0];
}

// Interleaved Pair Implementation with less divergence
__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n)
{
  // set thread ID
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // convert global data pointer to the local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x;

  // boundary check
  if (idx >= n)
    return;

  // in-place reduction in global memory
  for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
    if (tid < stride)
      idata[tid] += idata[tid+stride];

    __syncthreads();
  }

  // write result for this block to global memory
  if (tid == 0)
    g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned int n)
{
  // set thread ID
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  // convert global data pointer to the local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x * 2;
  // unrolling 2 data blocks
  if (idx + blockDim.x < n)
    g_idata[idx] += g_idata[idx + blockDim.x];
  __syncthreads();

  // in-place reduction in global memory
  for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
    if (tid < stride)
      idata[tid] += idata[tid +stride];

    // synchronize within threadblock
    __syncthreads();
  }

  // write result for this block to global memory
  if (tid == 0)
    g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling8(int *g_idata, int *g_odata, unsigned int n)
{
  // set thread ID
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

  // convert global data pointer to the local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x * 8;

  // unrolling 8
  if (idx + 7*blockDim.x < n) {
    int a1 = g_idata[idx];
    int a2 = g_idata[idx + blockDim.x];
    int a3 = g_idata[idx + 2*blockDim.x];
    int a4 = g_idata[idx + 3*blockDim.x];
    int a5 = g_idata[idx + 4*blockDim.x];
    int a6 = g_idata[idx + 5*blockDim.x];
    int a7 = g_idata[idx + 6*blockDim.x];
    int a8 = g_idata[idx + 7*blockDim.x];
    g_idata[idx] = a1+a2+a3+a4+a5+a6+a7+a8;
  }
  __syncthreads();

  // in-place reduction in global memory
  for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
    if (tid < stride)
      idata[tid] += idata[tid + stride];

    // synchronize within threadblock
    __syncthreads();
  }

  // write result for this block to global memory
  if (tid == 0)
    g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrollWarps8(int *g_idata, int *g_odata, unsigned int n)
{
  // set thread ID
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

  // convert global data pointer to the local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x * 8;

  // unrolling 8
  if (idx + 7*blockDim.x < n) {
    int a1 = g_idata[idx];
    int a2 = g_idata[idx + blockDim.x];
    int a3 = g_idata[idx + 2*blockDim.x];
    int a4 = g_idata[idx + 3*blockDim.x];
    int a5 = g_idata[idx + 4*blockDim.x];
    int a6 = g_idata[idx + 5*blockDim.x];
    int a7 = g_idata[idx + 6*blockDim.x];
    int a8 = g_idata[idx + 7*blockDim.x];
    g_idata[idx] = a1+a2+a3+a4+a5+a6+a7+a8;
  }
  __syncthreads();

  // in-place reduction in global memory
  for (int stride = blockDim.x/2; stride > 32; stride /= 2) {
    if (tid < stride)
      idata[tid] += idata[tid + stride];

    // synchronize within threadblock
    __syncthreads();
  }

  // unrolling warp
  if (tid < 32) {
    volatile int *vmem = idata;
    vmem[tid] += vmem[tid + 32];
    vmem[tid] += vmem[tid + 16];
    vmem[tid] += vmem[tid + 8];
    vmem[tid] += vmem[tid + 4];
    vmem[tid] += vmem[tid + 2];
    vmem[tid] += vmem[tid + 1];
  }

  // write result for this block to global memory
  if (tid == 0)
    g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceCompleteUnrollWarps8(int *g_idata, int *g_odata, unsigned int n)
{
  // set thread ID
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

  // convert global data pointer to the local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x * 8;

  // unrolling 8
  if (idx + 7*blockDim.x < n) {
    int a1 = g_idata[idx];
    int a2 = g_idata[idx + blockDim.x];
    int a3 = g_idata[idx + 2*blockDim.x];
    int a4 = g_idata[idx + 3*blockDim.x];
    int a5 = g_idata[idx + 4*blockDim.x];
    int a6 = g_idata[idx + 5*blockDim.x];
    int a7 = g_idata[idx + 6*blockDim.x];
    int a8 = g_idata[idx + 7*blockDim.x];
    g_idata[idx] = a1+a2+a3+a4+a5+a6+a7+a8;
  }
  __syncthreads();

  // in-place reduction and complete unroll
  if (blockDim.x >= 1024 && tid < 512)
    idata[tid] += idata[tid + 512];
  __syncthreads();

  if (blockDim.x >= 512 && tid < 256)
    idata[tid] += idata[tid + 256];
  __syncthreads();

  if (blockDim.x >= 256 && tid < 128)
    idata[tid] += idata[tid + 128];
  __syncthreads();

  if (blockDim.x >= 128 && tid < 64)
    idata[tid] += idata[tid + 64];
  __syncthreads();

  // unrolling warp
  if (tid < 32) {
    volatile int *vmem = idata;
    vmem[tid] += vmem[tid + 32];
    vmem[tid] += vmem[tid + 16];
    vmem[tid] += vmem[tid + 8];
    vmem[tid] += vmem[tid + 4];
    vmem[tid] += vmem[tid + 2];
    vmem[tid] += vmem[tid + 1];
  }

  // write result for this block to global memory
  if (tid == 0)
    g_odata[blockIdx.x] = idata[0];
}

template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata, unsigned int n)
{
  // set thread ID
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

  // convert global data pointer to the local pointer of this block
  int *idata = g_idata + blockIdx.x * blockDim.x * 8;

  // unrolling 8
  if (idx + 7*blockDim.x < n) {
    int a1 = g_idata[idx];
    int a2 = g_idata[idx + blockDim.x];
    int a3 = g_idata[idx + 2*blockDim.x];
    int a4 = g_idata[idx + 3*blockDim.x];
    int a5 = g_idata[idx + 4*blockDim.x];
    int a6 = g_idata[idx + 5*blockDim.x];
    int a7 = g_idata[idx + 6*blockDim.x];
    int a8 = g_idata[idx + 7*blockDim.x];
    g_idata[idx] = a1+a2+a3+a4+a5+a6+a7+a8;
  }
  __syncthreads();

  // in-place reduction and complete unroll
  if (iBlockSize >= 1024 && tid < 512)
    idata[tid] += idata[tid + 512];
  __syncthreads();

  if (iBlockSize >= 512 && tid < 256)
    idata[tid] += idata[tid + 256];
  __syncthreads();

  if (iBlockSize >= 256 && tid < 128)
    idata[tid] += idata[tid + 128];
  __syncthreads();

  if (iBlockSize >= 128 && tid < 64)
    idata[tid] += idata[tid + 64];
  __syncthreads();

  // unrolling warp
  if (tid < 32) {
    volatile int *vmem = idata;
    vmem[tid] += vmem[tid + 32];
    vmem[tid] += vmem[tid + 16];
    vmem[tid] += vmem[tid + 8];
    vmem[tid] += vmem[tid + 4];
    vmem[tid] += vmem[tid + 2];
    vmem[tid] += vmem[tid + 1];
  }

  // write result for this block to global memory
  if (tid == 0)
    g_odata[blockIdx.x] = idata[0];
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

  // kernel 2: reduceNeighboredLess
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  iStart = seconds();
  reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
  cudaDeviceSynchronize();
  iElaps = seconds() - iStart;
  cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++)
    gpu_sum += h_odata[i];
  printf("gpu NeighboredLess                  elapsed %8.4lf ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

  // kernel 3: reduceInterleaved
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  iStart = seconds();
  reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
  cudaDeviceSynchronize();
  iElaps = seconds() - iStart;
  cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x; i++)
    gpu_sum += h_odata[i];
  printf("gpu Interleaved                     elapsed %8.4lf ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

  // kernel 4: reduceUnrolling2
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  iStart = seconds();
  reduceUnrolling2<<<grid.x/2, block>>>(d_idata, d_odata, size);
  cudaDeviceSynchronize();
  iElaps = seconds() - iStart;
  cudaMemcpy(h_odata, d_odata, grid.x/2*sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x/2; i++)
    gpu_sum += h_odata[i];
  printf("gpu Unrolling2                      elapsed %8.4lf ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x/2, block.x);

  // kernel 5: reduceUnrolling8
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  iStart = seconds();
  reduceUnrolling8<<<grid.x/8, block>>>(d_idata, d_odata, size);
  cudaDeviceSynchronize();
  iElaps = seconds() - iStart;
  cudaMemcpy(h_odata, d_odata, grid.x/8*sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x/8; i++)
    gpu_sum += h_odata[i];
  printf("gpu Unrolling8                      elapsed %8.4lf ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x/8, block.x);

  // kernel 6: reduceUnrollWarps8
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  iStart = seconds();
  reduceUnrollWarps8<<<grid.x/8, block>>>(d_idata, d_odata, size);
  cudaDeviceSynchronize();
  iElaps = seconds() - iStart;
  cudaMemcpy(h_odata, d_odata, grid.x/8*sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x/8; i++)
    gpu_sum += h_odata[i];
  printf("gpu reduceUnrollWarps8              elapsed %8.4lf ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x/8, block.x);

  // kernel 7: reduceCompleteUnrollWarps8
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  iStart = seconds();
  reduceCompleteUnrollWarps8<<<grid.x/8, block>>>(d_idata, d_odata, size);
  cudaDeviceSynchronize();
  iElaps = seconds() - iStart;
  cudaMemcpy(h_odata, d_odata, grid.x/8*sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x/8; i++)
    gpu_sum += h_odata[i];
  printf("gpu reduceCompleteUnrollWarps8      elapsed %8.4lf ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x/8, block.x);

  // kernel 8: reduceCompleteUnroll
  cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  iStart = seconds();
  switch (blocksize) {
  case 1024:
    reduceCompleteUnroll<1024><<<grid.x/8, block>>>(d_idata, d_odata, size);
    break;
  case 512:
    reduceCompleteUnroll<512><<<grid.x/8, block>>>(d_idata, d_odata, size);
    break;
  case 256:
    reduceCompleteUnroll<256><<<grid.x/8, block>>>(d_idata, d_odata, size);
    break;
  case 128:
    reduceCompleteUnroll<128><<<grid.x/8, block>>>(d_idata, d_odata, size);
    break;
  case 64:
    reduceCompleteUnroll<64><<<grid.x/8, block>>>(d_idata, d_odata, size);
    break;
  }
  cudaDeviceSynchronize();
  iElaps = seconds() - iStart;
  cudaMemcpy(h_odata, d_odata, grid.x/8*sizeof(int), cudaMemcpyDeviceToHost);
  gpu_sum = 0;
  for (int i = 0; i < grid.x/8; i++)
    gpu_sum += h_odata[i];
  printf("gpu reduceCompleteUnroll            elapsed %8.4lf ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x/8, block.x);

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