//
// kernel.cu
//
// Copyright 2018 Makoto Shimazu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <assert.h>
#include <sys/time.h>

#include <string>

#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define check(f) assert(f == cudaSuccess)

namespace {

class Timer {
 public:
  Timer() : begin_ms_(now_ms()) {}
  void stop() { end_ms_ = now_ms(); }
  double elapsed_ms() const { return end_ms_ - begin_ms_; }

 private:
  double now_ms() const {
    timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1E3 + tv.tv_usec / 1E3;
  }

  double begin_ms_;
  double end_ms_;
};

}  // namespace

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void hgemm(half *a, half *b, float *c,
                      int M, int N, int K,
                      float alpha, float beta) {
  // Leading dimensions. Packed with no transpositions.
  const int lda = M;
  const int ldb = K;
  const int ldc = M;

  // Tile using a 2D grid
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  wmma::fill_fragment(acc_frag, 0.0f);

  // Loop over the K-dimension
  for (int i = 0; i < K; i += WMMA_K) {
    int aRow = warpM * WMMA_M;
    int aCol = i;
    int bRow = i;
    int bCol = warpN * WMMA_N;

    // Bounds checking
    if (aRow < M && aCol < K && bRow < K && bCol < N) {
      // Load the inputs
      wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
      wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

      // Perform the matrix multiplication
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }
  // Load in current value of c, scale by beta, and add to result scaled by alpha
  int cRow = warpM * WMMA_M;
  int cCol = warpN * WMMA_N;

  if (cRow < M && cCol < N) {
    wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);

    for(int i = 0; i < c_frag.num_elements; i++) {
      c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
    }

    // Store the output
    wmma::store_matrix_sync(c + cRow + cCol * ldc, acc_frag, ldc, wmma::mem_col_major);
  }
}

__global__ void init_matrixes(half* a, half* b, float* c, int M, int N, int K) {
  // Leading dimensions. Packed with no transpositions.
  const int lda = M;
  const int ldb = K;
  const int ldc = M;

  // Tile using a 2D grid
  int m = (blockIdx.x * blockDim.x + threadIdx.x);
  int n = (blockIdx.y * blockDim.y + threadIdx.y);
  if (m >= M || n >= N)
    return;

  for (int k = 0; k < K; ++k) {
    // a[m + k * lda] = __float2half(m + k * lda);
    a[m + k * lda] = __float2half(1.1f);
    b[k + n * ldb] = (k == n) ? __float2half(1.0f) :__float2half(0.0f);
  }

  c[m + n * ldc] = 0.0f;
}

__global__ void half2float(half* in, float* out, int M, int N) {
  // Tile using a 2D grid
  int m = (blockIdx.x * blockDim.x + threadIdx.x);
  int n = (blockIdx.y * blockDim.y + threadIdx.y);
  if (m >= M || n >= N)
    return;
  out[m + n * M] = __half2float(in[m + n * M]);
}

// Return milliseconds to run the kernel
__host__ double wmma_run(int N, int M, int K) {
  int devId;
  cudaDeviceProp props;
  check(cudaGetDeviceCount(&devId));
  check(cudaGetDeviceProperties(&props, devId - 1));
  printf("Device %d: \"%s\" with Compute %d.%d capability\n",
         devId, props.name, props.major, props.minor);

  const int num_threads_in_block = props.maxThreadsPerBlock;
  printf("Num of threads in a block: %d\n", num_threads_in_block);

  half* a, *b;
  float* c, *a_h, *b_h;

  // Allocate |a|, |B|, and |C|.
  check(cudaMalloc(reinterpret_cast<void **>(&a), sizeof(half) * M * K));
  check(cudaMalloc(reinterpret_cast<void **>(&b), sizeof(half) * K * N));
  check(cudaMallocManaged(reinterpret_cast<void **>(&c), sizeof(float) * M * N));

  check(cudaMallocManaged(reinterpret_cast<void **>(&a_h), sizeof(float) * M * K));
  check(cudaMallocManaged(reinterpret_cast<void **>(&b_h), sizeof(float) * K * N));

  check(cudaDeviceSynchronize());

  dim3 dimGrid(M / 32, N / 32);
  dim3 dimBlock(32, 32);

  init_matrixes<<<dimGrid, dimBlock>>>(a, b, c, M, N, K);
  check(cudaDeviceSynchronize());

  half2float<<<dimGrid, dimBlock>>>(a, a_h, M, K);
  half2float<<<dimGrid, dimBlock>>>(b, b_h, M, K);
  check(cudaDeviceSynchronize());

  printf(" === A ===\n");
  for (int m = 0; m < 10; m++) {
    for (int k = 0; k < 10; k++) {
      printf("%3.1f ", a_h[m + k * M]);
    }
    printf("\n");
  }

  printf(" === B ===\n");
  for (int k = 0; k < 10; k++) {
    for (int n = 0; n < 10; n++) {
      printf("%3.1f ", b_h[k + n * K]);
    }
    printf("\n");
  }

  Timer t;
  hgemm<<<dimGrid, dimBlock>>>(a, b, c, M, N, K, 1.0f, 1.0f);
  check(cudaDeviceSynchronize());
  t.stop();

  printf(" === A ===\n");
  for (int m = 0; m < 10; m++) {
    for (int k = 0; k < 10; k++) {
      printf("%3.1f ", a_h[m + k * M]);
    }
    printf("\n");
  }

  printf(" === B ===\n");
  for (int k = 0; k < 10; k++) {
    for (int n = 0; n < 10; n++) {
      printf("%3.1f ", b_h[k + n * K]);
    }
    printf("\n");
  }

  printf(" === C ===\n");
  for (int m = 0; m < 10; m++) {
    for (int n = 0; n < 10; n++) {
      printf("%3.1f ", c[m + n * M]);
    }
    printf("\n");
  }

  return t.elapsed_ms();
}