

// read .H5 file
//
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <hdf5/serial/H5Cpp.h>
#include <hdf5/serial/H5DataSet.h>
#include <hdf5/serial/H5DataSpace.h>
#include <hdf5/serial/H5DataType.h>
#include <hdf5/serial/H5Dpublic.h>
#include <hdf5/serial/H5Exception.h>
#include <hdf5/serial/H5File.h>
#include <hdf5/serial/H5Fpublic.h>
#include <hdf5/serial/H5PredType.h>
#include <hdf5/serial/H5public.h>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

template <typename DATA_T> struct H5TypeTraits {};

template <> struct H5TypeTraits<float> {
  static H5::DataType Type() { return H5::PredType::NATIVE_FLOAT; }
};

template <> struct H5TypeTraits<uint32_t> {
  static H5::DataType Type() { return H5::PredType::NATIVE_UINT32; }
};

class IOUitls {
public:
  explicit IOUitls(std::string filename) : filename_(std::move(filename)) {
    try {
      H5::Exception::dontPrint();
      file_ = H5::H5File(filename_, H5F_ACC_RDONLY);
    } catch (const H5::FileIException &error) {
      std::cerr << "Error opening file: " << "filename: " << filename_ << "\n "
                << error.getCDetailMsg() << std::endl;
      exit(1);
    }
  }
  ~IOUitls() { file_.close(); }

  template <typename DATA_T>
  std::vector<DATA_T> readDataset(const std::string &dataset_name,
                                  const H5::DataType dataType) {
    try {
      // open dataset
      auto dataset = file_.openDataSet(dataset_name);
      // open data space,get rank and dim of the dataset.
      auto dataset_space = dataset.getSpace();

      auto rank = dataset_space.getSimpleExtentNdims();
      std::vector<hsize_t> dims(rank);
      dataset_space.getSimpleExtentDims(dims.data(), nullptr);

      // get total size
      auto total = 1ll;
      for (auto &dim : dims) {
        total *= dim;
      }

      // read dataset
      std::vector<DATA_T> buffer(total);

      dataset.read(buffer.data(), dataType);
      return buffer;

    } catch (const H5::DataSetIException &error) {
      exit(1);
    } catch (const H5::DataSpaceIException &error) {
      exit(1);
    }
  }

private:
  std::string filename_;
  H5::H5File file_;
};