
#include <cassert>
#include <cstdint>
#include <sys/types.h>
template <typename DATA_T, uint32_t DATASET_DIM, uint32_t DATASET_SIZE>
class BaseIndex {
protected:
  bool is_vaild_;
  static const uint32_t VECTOR_BYTES = sizeof(DATA_T) * DATASET_DIM;
  static const uint32_t DATASET_BYTES = VECTOR_BYTES * DATASET_SIZE;

public:
  BaseIndex() : is_vaild_(false) {}

  virtual void construct(const DATA_T *vectors, const uint32_t n_vector) = 0;

  virtual void query(const DATA_T *query_vector, int topk, uint32_t *result) = 0;
};