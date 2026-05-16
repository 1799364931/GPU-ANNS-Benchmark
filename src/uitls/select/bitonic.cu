

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <functional>
#include <iostream>
#include <ostream>
#include <random>
#include <utility>
#include <vector>

#define SORT bitonicSortInWarp

template <typename T>
void generatorRandomArray(T *array, int size, int max_num) {
  auto rand = std::random_device{};
  auto ds = std::uniform_real_distribution<>(0, max_num);
  for (int i = 0; i < size; i++) {
    array[i] = ds(rand);
  }
}

inline bool isPowerTwo(int x) { return x > 0 && (x & (x - 1)) == 0; }

template <typename K, int N, bool Dir> __global__ void bitonicSort(K *k) {
  int idx = (threadIdx.x + blockDim.x * blockIdx.x);
  int kid = idx / N;
  int tid = idx % N;

  for (int l = 2; l <= N; l <<= 1) {
#pragma unroll
    for (int j = l >> 1; j > 0; j >>= 1) {
      int swap_id = tid ^ j;
      // 区分是上半还是下半
      bool low = !(tid & l);
      K ka = k[tid + N * kid];
      K kother = k[swap_id + N * kid];
      bool is_swap = (Dir ^ low) ? ka < kother : kother < ka;
      if (tid < swap_id && is_swap) {
        k[tid + N * kid] = kother;
        k[swap_id + N * kid] = ka;
      }
      __syncthreads();
    }
  }
}

template <typename K, int N, bool Dir> __global__ void bitonicSortOpt(K k[N]) {
  // 只用一半的线程
  for (int l = 2; l <= N; l <<= 1) {
#pragma unroll
    for (int j = l >> 1; j > 0; j >>= 1) {
      int pair_id = threadIdx.x;
      int tid = (pair_id / j) * (2 * j) + (pair_id % j);
      int swap_id = tid + j;

      // 区分是上半还是下半
      bool low = !(tid & l);
      bool is_swap = (Dir ^ low) ? k[tid] < k[swap_id] : k[swap_id] < k[tid];
      if (is_swap) {
        int t = k[tid];
        k[tid] = k[swap_id];
        k[swap_id] = t;
      }
    }
  }
}

// N must be power of 2
// and N <= warpsize ,blockdim == N
template <typename K, int N, bool Dir> __global__ void bitonicSortInWarp(K *k) {

  int lane_id = threadIdx.x % warpSize;
  int idx = (threadIdx.x + blockDim.x * blockIdx.x);
  int kid = idx / N;
  K Kt = k[lane_id + kid * N];
#pragma unroll
  for (int l = 2; l <= N; l <<= 1) {
    bool target_dir = (!Dir) ^ ((lane_id & l) != 0);
    // 如果是 1 方向就是 up
    // 如果是 2 方向就是 down
    for (int j = l >> 1; j > 0; j >>= 1) {
      K othert = __shfl_xor_sync(0xffff'ffff, Kt, j, N);
      // 需要判断 land_id 是不是低位线程
      bool low = !(lane_id & j);
      bool is_greater = Kt > othert;

      // 如果当前方向 和 目标方向不符
      if (target_dir == (is_greater ^ low)) {
        Kt = othert;
      }
    }
  }
  k[lane_id + kid * N] = Kt;
}

template <typename T, int N> bool isInOrder(bool Dir, T *array) {
  for (int i = 0; i < N - 1; i++) {
    if (Dir && array[i] > array[i + 1]) {
      return false;
    }
  }
  return true;
}

void kernelWrapper(std::function<void(cudaStream_t stream)> func) {
  cudaStream_t stream;
  cudaEvent_t start;
  cudaEvent_t end;

  cudaStreamCreate(&stream);
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start, stream);
  func(stream);
  cudaEventRecord(end, stream);

  cudaStreamSynchronize(stream);

  float ms;
  cudaEventElapsedTime(&ms, start, end);
  std::cout << "time:" << ms << "ms \n";

  cudaStreamDestroy(stream);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
}

int main() {
  constexpr int N = 32;
  int TEST_CNT = 100'00;
  int *nums = nullptr;

  cudaMallocHost(&nums, sizeof(int) * N * TEST_CNT);
  for (int i = 0; i < TEST_CNT; i++) {
    generatorRandomArray(nums + i * N, N, 1000);
  }

  auto sortFunc = [&](cudaStream_t stream) {
    SORT<int, N, false><<<TEST_CNT / 4, N * 4, 0, stream>>>(nums);
  };

  kernelWrapper(sortFunc);

  for (int i = 0; i < TEST_CNT; i++) {
    if (!isInOrder<int, N>(false, nums + i * N)) {
      std::cout << "error" << std::endl;
    }
  }

  cudaFree(nums);
}