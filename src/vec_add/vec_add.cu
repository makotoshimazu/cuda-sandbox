#include <cassert>
#include <cstdio>
#include <string>
#include <sys/time.h>

#define check(f) assert(f == cudaSuccess)

constexpr int N = 1024 * 1024 * 1024;
constexpr int STRIDE = 1024;

class Timer {
 public:
  Timer(std::string tag) : tag_(std::move(tag)) {
    timeval tv;
    gettimeofday(&tv, nullptr);
    begin_ms_ = tv.tv_sec * 1E3 + tv.tv_usec / 1E3;
  }

  ~Timer() {
    timeval tv;
    gettimeofday(&tv, nullptr);
    printf("[%s] %3.2f ms\n",
           tag_.c_str(),
           (tv.tv_sec * 1E3 + tv.tv_usec / 1E3) - begin_ms_);
  }

 private:
  std::string tag_;
  double begin_ms_;
};

__global__ void vec_add(float* a, float* b, float *c) {
  const int base = (blockIdx.x * blockDim.x + threadIdx.x) * STRIDE;
  for (int i = base; i < base + STRIDE; ++i) {
    c[i] = a[i] + b[i];
  }
}

__global__ void init(float* a, float* b, float* c) {
  const int base = (blockIdx.x * blockDim.x + threadIdx.x) * STRIDE;
  for (int i = base; i < base + STRIDE; ++i) {
    a[i] = i;
    b[i] = 1.0f;
    c[i] = 0.0f;
  }
}

int main() {
  int devId;
  cudaDeviceProp props;
  check(cudaGetDeviceCount(&devId));
  check(cudaGetDeviceProperties(&props, devId - 1));
  printf("Device %d: \"%s\" with Compute %d.%d capability\n",
         devId, props.name, props.major, props.minor);

  const int num_threads_in_block = props.maxThreadsPerBlock;
  printf("Num of threads in a block: %d\n", num_threads_in_block);

  float* A, *B, *C, *D;

  // Allocate |A|, |B|, and |C|.
  check(cudaMallocManaged(reinterpret_cast<void **>(&A), sizeof(float) * N));
  check(cudaMallocManaged(reinterpret_cast<void **>(&B), sizeof(float) * N));
  check(cudaMallocManaged(reinterpret_cast<void **>(&C), sizeof(float) * N));
  check(cudaMallocManaged(reinterpret_cast<void **>(&D), sizeof(float) * N));

  // Determine grid size and block size. They are single dimension.
  dim3 dimGrid(N / num_threads_in_block / STRIDE);
  dim3 dimBlock(num_threads_in_block);

  // Initialize vectors.
  {
    Timer t("init");
    init<<<dimGrid, dimBlock>>>(A, B, C);
    check(cudaDeviceSynchronize());
  }

  // Calculate vectors.
  {
    Timer t("vec_add (GPU)");
    vec_add<<<dimGrid, dimBlock>>>(A, B, C);
    check(cudaDeviceSynchronize());
  }

  {
    Timer t("vec_add (CPU)");
    for (int i = 0; i < N; ++i)
      D[i] = A[i] + B[i];
  }

  // Verify the answers.
  int wrong_num = 0;
  int wrong_pos = -1;
  for (int i = 0; i < N; ++i) {
    if (C[i] != D[i]) {
      wrong_num++;
      if (wrong_pos < 0)
        wrong_pos = i;
    }
  }
  if (wrong_num) {
    printf("wrong_num: %d, wrong_pos: %d\n", wrong_num, wrong_pos);
    printf("[%d] %3.1f + %3.1f = (%3.1f, %3.1f)\n", wrong_pos, A[wrong_pos], B[wrong_pos], C[wrong_pos], D[wrong_pos]);
  } else {
    printf("Good!\n");
  }

  check(cudaFree(A));
  check(cudaFree(B));
  check(cudaFree(C));
  return 0;
}