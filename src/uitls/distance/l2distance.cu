

// input_a  (batch × dim)
// input_b (dim × batch)
// output (batch × batch)

// dim / TILE_WIDTH == batch / TILE_HEIGHT
#include <iostream>
#include <random>
template <typename T, int TILE_WIDTH, int TILE_HEIGHT>
__global__ void L2Distance(T *input_a, T *input_b, T *output, int batch,
                           int dim) {
  __shared__ T tile_a[TILE_HEIGHT][TILE_WIDTH];
  __shared__ T tile_b[TILE_HEIGHT][TILE_WIDTH];

  int tx = threadIdx.x, ty = threadIdx.y;
  int bx = blockIdx.x, by = blockIdx.y;

  int col = bx * TILE_WIDTH + tx;
  int row = by * TILE_HEIGHT + ty;
  T val = 0;

  for (int ph = 0; ph < dim / TILE_WIDTH; ph++) {
    tile_a[ty][tx] = input_a[row * dim + (tx + ph * TILE_WIDTH)];
    tile_b[ty][tx] = input_b[(ty + ph * TILE_WIDTH) * batch + col];

    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; k++) {
      T temp = tile_a[ty][k] - tile_b[k][tx];
      val += sqrt(temp * temp);
    }
    __syncthreads();
  }
  output[row * batch + col] = val;
}

template <typename T>
void generatorRandomArray(T *array, int size, int max_num) {
  auto rand = std::random_device{};
  auto ds = std::uniform_real_distribution<>(0, max_num);
  for (int i = 0; i < size; i++) {
    array[i] = ds(rand);
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

int main() {
  // batch
  int batch = 4, dim = 4;
  int *input_a = nullptr, *input_b = nullptr, *output = nullptr;
  cudaMallocManaged(&input_a, sizeof(int) * (batch * dim));
  cudaMallocManaged(&input_b, sizeof(int) * (batch * dim));
  cudaMallocManaged(&output, sizeof(int) * (batch * batch));
  generatorRandomArray(input_a, batch * dim, 10);
  generatorRandomArray(input_b, batch * dim, 10);

  print(input_a, batch, dim);
  print(input_b, dim, batch);
  // 一个线程对应一个元素 batch * batch
  dim3 grid(batch, batch, 1);
  dim3 block(dim, batch, 1);
  L2Distance<int, 4, 4><<<1, block>>>(input_a, input_b, output, batch, dim);
  cudaDeviceSynchronize();
  print(output, batch, batch);
  // 25 + 9 +36 + 1
  cudaFree(input_a);
  cudaFree(input_b);
  cudaFree(output);
}
