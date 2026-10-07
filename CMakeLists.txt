cmake_minimum_required(VERSION 3.18)
project(muscl_step LANGUAGES CXX)

# Configure with the Intel oneAPI compilers:
#   cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx ..
# On NVIDIA GPUs (needs the oneAPI plugin for NVIDIA GPUs) add:
#   -DSYCL_TARGETS=nvptx64-nvidia-cuda

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(SYCL_TARGETS "" CACHE STRING "Optional -fsycl-targets value, e.g. nvptx64-nvidia-cuda")

# Find pybind11 through the active Python environment (pip install pybind11)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE PYBIND11_FOUND_RESULT)
if(NOT PYBIND11_FOUND_RESULT EQUAL 0)
  message(FATAL_ERROR "pybind11 not found: run 'pip install pybind11'")
endif()
find_package(pybind11 CONFIG REQUIRED)

# Python extension module: import muscl_step
pybind11_add_module(muscl_step full_muscl_solver.cpp)

set(SYCL_FLAGS -fsycl)
if(SYCL_TARGETS)
  list(APPEND SYCL_FLAGS -fsycl-targets=${SYCL_TARGETS})
endif()

target_compile_options(muscl_step PRIVATE ${SYCL_FLAGS} -O3)
target_link_options(muscl_step PRIVATE ${SYCL_FLAGS})
