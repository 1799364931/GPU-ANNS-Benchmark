#include "/usr/GPU/GPU-ANNS-Benchmark/src/uitls/dataset/IOUitls.hpp"
#include "index.cuh"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
using namespace burst_force;
#define SIFT_DATASET_DIM 128
#define SIFT_DATASET_SIZE 1'000'000
#define SIFT_QUERY_DIM 100
int main() {
  BurstForce<float, SIFT_DATASET_DIM, SIFT_DATASET_SIZE> index;
  IOUitls file("/usr/SGX_LSH/sgx_mvp/datafile/sift-128-euclidean.hdf5");

  // load train
  auto train = file.readDataset<float>("train", H5TypeTraits<float>::Type());

  // load query
  auto test = file.readDataset<float>("test", H5TypeTraits<float>::Type());

  // load match
  auto neighbors =
      file.readDataset<uint32_t>("neighbors", H5TypeTraits<uint32_t>::Type());

  std::cout << "load data success\n";

  index.construct(train.data(), SIFT_DATASET_SIZE);

  std::cout << "construct success\n";

  float avg_recallk = 0;
  for (int i = 0; i < 1000; i++) {
    std::vector<uint32_t> result(SIFT_QUERY_DIM);
    index.query(test.data() + i * SIFT_DATASET_DIM, SIFT_QUERY_DIM,
                result.data());
    // calculate recall topk
    int hit = 0;
    for (int k = 0; k < SIFT_QUERY_DIM; k++) {
      for (int j = 0; j < SIFT_QUERY_DIM; j++) {
        if (result[j] == neighbors[k + i * SIFT_QUERY_DIM]) {
          hit++;
          break;
        }
      }
    }
    avg_recallk += 1.0 * hit / SIFT_QUERY_DIM;
  }
  std::cout << avg_recallk << "\n";
}