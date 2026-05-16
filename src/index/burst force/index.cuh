

#include "../BaseIndex.hpp"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <utility>

namespace burst_force {

// 1. blockdim.x >= 128
// 2. topk <= 100

// todo debug this function
template <typename DATA_T, class IDX_T>
void __global__ search(const DATA_T *query, const uint32_t top_k,
                       const IDX_T *index, const uint32_t dataset_dim,
                       const uint32_t dataset_size, uint32_t *result) {

  __shared__ int s_candidate_size;
  __shared__ int s_candidate_max_size;
  extern __shared__ unsigned char s_mem[];
  DATA_T *s_query = reinterpret_cast<DATA_T *>(s_mem);
  std::pair<float, uint32_t> *s_candidate =
      reinterpret_cast<std::pair<float, uint32_t> *>(s_query + dataset_dim);

  // init shared memory
  if (threadIdx.x == 0) {
    s_candidate_size = 0;
    s_candidate_max_size = 128;
  }

  // load query vector
  for (int start = 0; start < dataset_dim; start += blockDim.x) {
    int idx = start + threadIdx.x;
    if (idx < dataset_dim) {
      s_query[idx] = query[idx];
    }
  }
  __syncthreads();

  // init s_candidate
  for (int start = 0; start < s_candidate_max_size; start += blockDim.x) {
    int idx = start + threadIdx.x;
    if (idx < s_candidate_max_size) {
      s_candidate[idx].first = MAXFLOAT;
      s_candidate[idx].second = UINT32_MAX;
    }
  }
  __syncthreads();

  // 32 threads process one vector
  float diff = 0;
  int warp_num = blockDim.x / warpSize;
  int warp_id = threadIdx.x / warpSize;
  int lane_id = threadIdx.x % warpSize;

  for (int v_start = 0; v_start < dataset_size; v_start += warp_num) {
    int vector_id = v_start + warp_id;

    for (int i_start = 0; i_start < dataset_dim; i_start += warpSize) {
      float temp = 0;
      if (i_start + lane_id < dataset_dim) {
        temp = (s_query[i_start] -
                index[vector_id * dataset_dim + i_start + lane_id]);
      }

      diff += (temp * temp);
    }

#pragma unroll
    for (int i = 1; i < warpSize; i <<= 1) {
      diff += __shfl_down_sync(0xffffffff, diff, 1);
    }

    if (lane_id == 0) {
      int pos = atomicAdd(&s_candidate_size, 1);
      s_candidate[pos].first = diff;
      s_candidate[pos].second = vector_id;
    }
    __syncthreads();

    // once loop produce 'blockdim.x / warpSize' new candidate
    // at most 32 new candidate
    // just sort them and insert to the candidate

    // at first , we assume blockdim.x >= s_candidate_size. and s_candidate_size
    // must be power of 2? todo , to deal blockdim.x < s_candidate_size
    // situation.
    if (threadIdx.x < s_candidate_size) {
      int tid = threadIdx.x;
      // sort
      for (int k = 2; k <= s_candidate_max_size; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
          int ixj = tid ^ j;
          if (ixj > tid) {
            bool up = ((tid & k) == 0);
            if (s_candidate[tid].first > s_candidate[ixj].first == up) {
              auto temp = s_candidate[tid];
              s_candidate[tid].first = s_candidate[ixj].first;
              s_candidate[tid].second = s_candidate[ixj].second;
              s_candidate[ixj].first = temp.first;
              s_candidate[ixj].second = temp.second;
            }
          }
          __syncthreads();
        }

      // update candidate size
      if (threadIdx.x == 0) {
        if (s_candidate_size > top_k) {
          s_candidate_size = top_k;
        }
      }
    }

    // submit to the result
    if (threadIdx.x < top_k) {
      result[threadIdx.x] = s_candidate[threadIdx.x].second;
    }
  }
}

template <typename DATA_T, uint32_t DATASET_DIM, uint32_t DATASET_SIZE>
class BurstForce : BaseIndex<DATA_T, DATASET_DIM, DATASET_SIZE> {

  DATA_T *index_;

public:
  BurstForce() : index_(nullptr) {};
  ~BurstForce() {
    if (index_ != nullptr) {
      std::free(index_);
    }
    this->is_vaild_ = false;
  }

  // delete copy/assign construct function
  BurstForce(const BurstForce<DATA_T, DATASET_DIM, DATASET_SIZE> &other) =
      delete;
  BurstForce<DATA_T, DATASET_DIM, DATASET_SIZE> &operator=(
      const BurstForce<DATA_T, DATASET_DIM, DATASET_SIZE> &other) = delete;

  BurstForce(BurstForce<DATA_T, DATASET_DIM, DATASET_SIZE> &&other) {
    index_ = other.index_;
    this->is_vaild_ = true;
    other.is_vaild_ = false;
    other.index_ = nullptr;
  }

  BurstForce<DATA_T, DATASET_DIM, DATASET_SIZE> &
  operator=(const BurstForce<DATA_T, DATASET_DIM, DATASET_SIZE> &&other) {
    if (*other == this) {
      return *this;
    }
    ~BurstForce();
    index_ = other.index_;
    this->is_vaild_ = true;
    other.is_vaild_ = false;
    other.index_ = nullptr;
    return *this;
  }

  void construct(const DATA_T *vectors, const uint32_t n_vector) override {
    assert(!this->is_vaild_ && index_ == nullptr &&
           "index can't be construct twice.");
    std::cout << " test\n";
    // load data
    index_ = static_cast<DATA_T *>(std::malloc(this->DATASET_BYTES));
    std::cout << "DATASET_BYTES = " << this->DATASET_BYTES << "\n";
    std::cout << "index_ = " << index_ << "\n";
    std::cout << "vectors = " << vectors << "\n";
    assert(index_ != nullptr && "index malloc fail.");
    std::memcpy(index_, vectors, this->DATASET_BYTES);
    std::cout << " test\n";
    this->is_vaild_ = true;
  }

  void query(const DATA_T *query_vector, int topk, uint32_t *result) override {
    assert(this->is_vaild_ && "index must be construct before query.");
    assert(query_vector && result);
    // todo add cuda error
    DATA_T *index_d;
    uint32_t *result_d;
    cudaMalloc(&index_d, this->DATASET_BYTES);
    cudaMalloc(&result_d, topk * sizeof(uint32_t));
    result = static_cast<uint32_t *>(std::malloc(topk * sizeof(uint32_t)));
    // start query
    dim3 block(256, 1, 1);

    search<<<1, block, 1024>>>(query_vector, topk, index_, DATASET_DIM,
                                  DATASET_SIZE, result_d);
    cudaDeviceSynchronize();

    cudaMemcpy(result, result_d, sizeof(uint32_t) * topk,
               cudaMemcpyDeviceToHost);

    cudaFree(result_d);
    cudaFree(index_d);
  }
};

}; // namespace burst_force