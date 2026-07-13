cmake_minimum_required(VERSION 3.14)
project(accorder_perm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# OpenMP
find_package(OpenMP REQUIRED)

# Boost (header-only components used by rabbit_order / my_order)
find_package(Boost REQUIRED)

# numa
find_library(NUMA_LIB numa REQUIRED)

add_executable(accorder_perm accorder_perm.cpp)

target_include_directories(accorder_perm PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${Boost_INCLUDE_DIRS}
)

target_link_libraries(accorder_perm PRIVATE
  OpenMP::OpenMP_CXX
  ${NUMA_LIB}
)
