
#include <cassert>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iostream>
#include <random>
namespace uitls {

// provide some random function
class Random {
public:
  template <typename T>
  static inline void generatorRandomList(T *array, std::string file_path,
                                         int length, int max, int threadcount = 10) {
    assert(array);
    auto rand = std::random_device{};
    auto ds = std::uniform_real_distribution<>(0, max);
    for (int i = 0; i < length; i++) {
      array[i] = ds(rand);
    }

    // save to file
    // if file not exist create
    std::fstream file(file_path, std::ios::out);
    for (int i = 0; i < length; i++) {
      file << array[i] << " ";
    }
    file.close();
    return;
  }

  template <typename T>
  static inline bool generatorRandomListFromFile(T *array,
                                                 std::string file_path,
                                                 int length, int max) {
    assert(array);

    std::fstream file(file_path, std::ios::in);
    if (!file.is_open()) {
      return false;
    }

    for (int i = 0; i < length; i++) {
      file >> array[i];
    }
    file.close();

    return true;
  }
};

} // namespace uitls