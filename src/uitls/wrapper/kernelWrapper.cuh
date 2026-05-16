
#include <functional>
struct KernelWrapper {

  cudaStream_t stream;
  cudaEvent_t start;
  cudaEvent_t end;

  float operator()(std::function<void(cudaStream_t stream)> func) {

    cudaStreamCreate(&stream);
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start, stream);
    func(stream);
    cudaEventRecord(end, stream);

    cudaStreamSynchronize(stream);

    float ms;
    cudaEventElapsedTime(&ms, start, end);

    return ms;
  }

  KernelWrapper() {}

  ~KernelWrapper() {
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
  }
};