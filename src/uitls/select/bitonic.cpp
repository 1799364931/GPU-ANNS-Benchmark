

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

bool isPowerTwo(int x) { return x > 0 && (x & (x - 1)) == 0; }

// 让一个双调序列变有序
template <typename K, bool Dir>
void bitnoicMergeStep(std::vector<K> &k, int left, int right) {
  int N = right - left;
  if (N == 1) {
    return;
  }
  assert(isPowerTwo(N));

  for (int i = left; i < left + N / 2; i++) {
    bool is_swap = Dir ? k[i] > k[i + N / 2] : k[i] < k[i + N / 2];
    if (is_swap) {
      std::swap(k[i], k[i + N / 2]);
    }
  }

  bitnoicMergeStep<K, Dir>(k, left, left + N / 2);
  bitnoicMergeStep<K, Dir>(k, left + N / 2, right);
}

template <typename K, bool Dir>
void bitnoicSortStep(std::vector<K> &k, int left, int right) {
  int N = right - left;
  if (N == 1) {
    return;
  }
  assert(isPowerTwo(N));

  bitnoicSortStep<K, true>(k, left, left + N / 2);
  bitnoicSortStep<K, false>(k, left + N / 2, right);

  bitnoicMergeStep<K, Dir>(k, left, right);
}

int main() {
  std::vector<int> nums = {22, 24, 49, 50, 40, 33, 21, 11};
  bitnoicSortStep<int, true>(nums, 0, nums.size());
  std::for_each(nums.begin(), nums.end(),
                [](int num) { std::cout << num << " "; });
  std::cout << "\n";
}