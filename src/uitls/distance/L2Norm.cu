

#include "/usr/GPU-ANNS-Benchmark/src/uitls/random/random.hpp"
#include "/usr/GPU-ANNS-Benchmark/src/uitls/wrapper/kernelWrapper.cuh"

#include <cmath>
#include <cstdio>

#include <iostream>

template <typename T, int TILE_SIZE>
__global__ void L2Norm(T *input, T *output, int batch, int dim) {
  // 最简单的算法
  extern __shared__ float smem[]; // TILE_SIZE

  int row_start = TILE_SIZE * blockIdx.x;
  bool is_last_tile = (blockIdx.x == (gridDim.x - 1));

  if (is_last_tile) {
    if (threadIdx.x < batch - row_start) {
      smem[threadIdx.x] = 0;
    }
    __syncthreads();

    // 错误写法
    // for (int row = 0; row < batch - row_start; row++) {
    //   for (int col = threadIdx.x; col < dim; col += blockDim.x) {
    //     T tmp = input[(row_start + row) * dim + col];
    //     // 同步
    //     atomicAdd(smem + (row + row_start), tmp * tmp);
    //     // 这也很坑，尽量别在 if 语句做同步。

    //     __syncthreads();
    //   }
    // }

    for (int row = 0; row < batch - row_start; row++) {
      float norm = 0;
#pragma unroll
      for (int col = threadIdx.x; col < dim; col += blockDim.x) {
        T tmp = input[(row_start + row) * dim + col];
        norm += tmp * tmp;
      }
      atomicAdd(smem + row, norm);
    }

    __syncthreads();
    if (threadIdx.x < batch - row_start) {
      output[row_start + threadIdx.x] = smem[threadIdx.x];
    }

  } else {

    if (threadIdx.x < TILE_SIZE) {
      smem[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int row = 0; row < TILE_SIZE; row++) {
      float norm = 0;
#pragma unroll
      for (int col = threadIdx.x; col < dim; col += blockDim.x) {
        T tmp = input[(row_start + row) * dim + col];
        norm += tmp * tmp;
      }
      atomicAdd(smem + row, norm);
    }

    __syncthreads();
    if (threadIdx.x < TILE_SIZE) {
      output[row_start + threadIdx.x] = smem[threadIdx.x];
    }
  }
}

// input (batch,dim)
// output (batch)
template <typename T, int TILE_SIZE>
__global__ void L2NormOpt(T *input, T *output, int batch, int dim) {
  // thread -> warp -> block

  extern __shared__ float smem[]; // TILE_SIZE * WARPNUM

  // blockDim.x must div warpSize

  // how many warp in a block
  int num_warp = blockDim.x / warpSize;
  bool is_last_tile = (blockIdx.x == (gridDim.x - 1));

  int row_start = TILE_SIZE * blockIdx.x;

  float row_norm[TILE_SIZE];
  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;

  if (is_last_tile) {
    // 逐个相加
    for (int row = 0; row < batch - row_start; row++) {
      row_norm[0] = 0;
#pragma unroll
      for (int col = threadIdx.x; col < dim; col += blockDim.x) {
        float tmp = input[(row + row_start) * dim + col];
        row_norm[0] += tmp * tmp;
      }
#pragma unroll
      for (int w = 1; w < warpSize; w <<= 1) {
        row_norm[0] += __shfl_down_sync(0xffff'ffff, row_norm[0], w, warpSize);
      }

      if (lane_id == 0) {
        smem[num_warp * row + warp_id] = row_norm[0];
      }
    }
  } else {
    float tmp[TILE_SIZE];
    for (int i = 0; i < TILE_SIZE; i++) {
      row_norm[i] = 0;
    }

    for (int col = threadIdx.x; col < dim; col += blockDim.x) {

#pragma unroll
      for (int row = 0; row < TILE_SIZE; row++) {
        tmp[row] = input[(row + row_start) * dim + col];
      }
#pragma unroll
      for (int row = 0; row < TILE_SIZE; row++) {
        tmp[row] = (tmp[row] * tmp[row]);
      }
#pragma unroll
      for (int row = 0; row < TILE_SIZE; row++) {
        row_norm[row] += tmp[row];
      }
    }

// reduce
#pragma unroll
    // 64 -> 2 warp -> block
    for (int row = 0; row < TILE_SIZE; row++) {
#pragma unroll
      for (int w = 1; w < warpSize; w <<= 1) {
        row_norm[row] +=
            __shfl_down_sync(0xffff'ffff, row_norm[row], w, warpSize);
      }

      if (lane_id == 0) {
        smem[num_warp * row + warp_id] = row_norm[row];
      }
    }
  }

  __syncthreads();
  // 49 + 64 = 117
  if (warp_id == 0) {
#pragma unroll
    for (int row = 0; row < TILE_SIZE; row++) {
      row_norm[row] = lane_id < num_warp ? smem[num_warp * row + lane_id] : 0;
    }
#pragma unroll
    for (int row = 0; row < TILE_SIZE; row++) {

#pragma unroll
      for (int w = 1; w < warpSize; w <<= 1) {
        row_norm[row] +=
            __shfl_down_sync(0xffff'ffff, row_norm[row], w, warpSize);
      }

      if (lane_id == 0) {
        output[row + row_start] = row_norm[row];
      }
    }
  }
}

template <typename T> void print(T *array, int height, int width) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      std::cout << array[i * width + j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

template <typename T> bool check(T *input, T *output, int batch, int dim) {
  for (int i = 0; i < batch; i++) {
    float l2norm = 0;
    for (int j = 0; j < dim; j++) {
      l2norm += input[i * dim + j] * input[i * dim + j];
    }
    if (output[i] != l2norm) {
      // std::cout << l2norm << " " << output[i] << "\n";
      return false;
    }
  }
  return true;
}

int main() {

  constexpr int batch = 1 << 12, dim = 1 << 16, TILE_SIZE = 8;
  int *input = nullptr, *output = nullptr;
  cudaMallocManaged(&input, sizeof(int) * (batch * dim));
  cudaMallocManaged(&output, sizeof(int) * batch);
  if (!uitls::Random::generatorRandomListFromFile(input, "./data.txt",
                                                  (batch * dim), 10)) {
    uitls::Random::generatorRandomList(input, "./data.txt", (batch * dim), 10);
  }

  std::cout << "data init success\n";

  dim3 block(256);
  dim3 grid(batch / TILE_SIZE);

  // print(input, batch, dim);

  KernelWrapper kw;
  auto res = kw([&](cudaStream_t stream) {
    int device_id;
    cudaGetDevice(&device_id);
    cudaMemPrefetchAsync(input, batch * dim * sizeof(int), device_id, stream);
    cudaMemPrefetchAsync(output, batch * sizeof(int), device_id, stream);

#ifdef OPT
    L2NormOpt<int, TILE_SIZE>
        <<<grid, block, TILE_SIZE *(block.x / 32) * sizeof(float), stream>>>(
            input, output, batch, dim);
#else
    L2Norm<int, TILE_SIZE><<<grid, block, TILE_SIZE * sizeof(float), stream>>>(
        input, output, batch, dim);
#endif
  });

  if (!check(input, output, batch, dim)) {
    std::cout << "error" << "\n";
  }

  std::cout << "time:" << res << "\n";

  // print(output, 1, batch);
  cudaFree(input);
  cudaFree(output);
}