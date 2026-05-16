
cmake_minimum_required(VERSION 3.10)
project(test LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set cuda as compiler
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

add_executable(test indexTest.cu)

find_package(HDF5 REQUIRED COMPONENTS CXX)
target_link_libraries(test PRIVATE 
    HDF5::HDF5
)


set_target_properties(test PROPERTIES LINKER_LANGUAGE CUDA)