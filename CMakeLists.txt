cmake_minimum_required(VERSION 3.5)
project(fast_wmd)

set(CMAKE_CXX_STANDARD 11)

add_executable(fast_wmd src/main.cpp src/Tools.cpp src/Tools.hpp src/Clustering.cpp src/Clustering.hpp src/Distances.cpp src/Distances.hpp)

# Add ortools
find_library(ORTOOLS_LIB ortools)
target_link_libraries(fast_wmd "${ORTOOLS_LIB}")

# Enable parallelization
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# Disable vectorize for fair comparison between distance methods
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-tree-vectorize -DEIGEN_DONT_VECTORIZE=1")

# Add optimization parameter
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")