cmake_minimum_required(VERSION 3.12.0)
# if we need -fPIC globally
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Set build type to Release by default
set(CMAKE_BUILD_TYPE RELEASE)
if (NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELEASE)
endif()

set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)
#set(CMAKE_CUDA_COMPILER /hpc/software/spack/opt/spack/linux-rhel7-x86_64/intel-19.0.5.281/cuda-11.2.1-5z4nepga7cson2vi56m4x7ocghdigzp4/bin/nvcc)

project(voxcraft-sim LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
################################################################
# Configuration for depended libraries
################################################################
set(Boost_USE_STATIC_LIBS   ON)
find_package(Boost REQUIRED COMPONENTS filesystem program_options)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(lib/pybind11)

# CUDA options
find_package(CUDA) # for CUDA_SELECT_NVCC_ARCH_FLAGS function
# if one want to force use certain Gpu Architecture, one can add `CUDA_ARCH` like `cmake -DCUDA_ARCH=arch=compute_61,code=sm_61 ..`
if (NOT DEFINED CUDA_ARCH)
  CUDA_SELECT_NVCC_ARCH_FLAGS(RET_ARCH_FLAGS)
  message("CUDA_SELECT_NVCC_ARCH_FLAGS returns: " ${RET_ARCH_FLAGS})
  list(LENGTH RET_ARCH_FLAGS FLAG_LENGTH)
  message("Flag length: " ${FLAG_LENGTH})
  if (FLAG_LENGTH GREATER 3)
    # CUDA_SELECT_NVCC_ARCH_FLAGS resulted more than 3 items, which is probably GPU not found error.
    message(STATUS "Nvidia GPU not found. Proceed with caution!")
  else()
    list(GET RET_ARCH_FLAGS -1 CUDA_ARCH) # use newest architecture
    message("Using GPU Architecture: " ${CUDA_ARCH})
    string(REGEX MATCH "compute_([0-9]+),*" TMP "${CUDA_ARCH}")
    message("Matched Compute Architecture: " ${CMAKE_MATCH_1})
    set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_MATCH_1})
    set(ARCH_FLAGS "-gencode" ${CUDA_ARCH})
    list(APPEND CMAKE_CUDA_FLAGS -gencode=${CUDA_ARCH})
  endif()
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if(uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  if (NOT DEFINED CUDA_DEBUG)
    set(CUDA_DEBUG "-G or not" ON)
  endif()
endif(uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
#-MG means debug info in GPU, so you can step in GPU kernels
if (CUDA_DEBUG)
  message("Set CUDA_DEBUG to ON.")
  set(CUDA_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:-G ${ARCH_FLAGS} --expt-relaxed-constexpr>)
else (CUDA_DEBUG)
  message("Set CUDA_DEBUG to OFF.")
  set(CUDA_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>: ${ARCH_FLAGS} --expt-relaxed-constexpr>)
endif(CUDA_DEBUG)
################################################################
# Main builds
################################################################
include_directories(lib)
include_directories(src)
include_directories(src/vx3)
include_directories(src/utils)
include_directories(src/vxa)

file(GLOB UTILS_SRC_cu src/utils/*.cu)
file(GLOB VX3_SRC_cu src/vx3/*.cu)
file(GLOB VXA_SRC_cu src/vxa/*.cu)
file(GLOB PYTHON_SRC_cu src/python/*.cu)

add_library(voxelyze3 ${UTILS_SRC_cu} ${VX3_SRC_cu} ${VXA_SRC_cu})
target_link_libraries(voxelyze3 PUBLIC ${Boost_LIBRARIES})
set_target_properties(voxelyze3 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(voxelyze3 PUBLIC ${CUDA_OPTIONS})

add_executable(voxcraft-sim src/executables/voxcraft_sim.cu)
target_link_libraries(voxcraft-sim PUBLIC voxelyze3)
target_link_libraries(voxcraft-sim PUBLIC ${Boost_LIBRARIES})
set_target_properties(voxcraft-sim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(voxcraft-sim PUBLIC ${CUDA_OPTIONS})

add_executable(voxcraft-test src/executables/test.cu)
target_link_libraries(voxcraft-test PUBLIC voxelyze3)
target_link_libraries(voxcraft-test PUBLIC ${Boost_LIBRARIES})
set_target_properties(voxcraft-test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(voxcraft-test PUBLIC ${CUDA_OPTIONS})

pybind11_add_module(voxcraft SHARED ${PYTHON_SRC_cu})
target_link_libraries(voxcraft PRIVATE
        ${Python3_LIBRARIES}
        voxelyze3)
target_include_directories(
        voxcraft PUBLIC
        ${Python3_INCLUDE_DIRS}
        ${PYBIND11_INCLUDE_DIR})
install(TARGETS voxcraft LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
################################################################
# Testing
################################################################
#option(test "Build all tests." OFF) # Makes boolean 'test' available.
#if (test)
#  if (APPLE)
#    add_definitions(-DGTEST_USE_OWN_TR1_TUPLE)
#    add_definitions(-D__GLIBCXX__)
#  endif (APPLE)
#
#  # This adds another subdirectory, which has 'project(gtest)'.
#  add_subdirectory(test/googletest)
#
#  enable_testing()
#
#  # Include the gtest library. gtest_SOURCE_DIR is available due to
#  # 'project(gtest)' above.
#  include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
#
#  ##############
#  # Unit Tests
#  ##############
#  set(TEST_DIR test/testcase)
#  file(GLOB TEST_FILES ${TEST_DIR}/test_*.c*)
#  add_executable(runUnitTests ${TEST_FILES})
#
#  # Standard linking to gtest stuff.
#  target_link_libraries(runUnitTests gtest gtest_main)
#
#  # Extra linking for the project.
#  target_link_libraries(runUnitTests voxelyze.3.0)
#
#  # This is so you can do 'make test' to see all your tests run, instead of
#  # manually running the executable runUnitTests to see those specific tests.
#  add_test(NAME that-test-I-made COMMAND runUnitTests)
#
#  # You can also omit NAME and COMMAND. The second argument could be some other
#  # test executable.
#  add_test(that-other-test-I-made runUnitTests)
#
#  set_target_properties(runUnitTests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#
#  target_compile_options(runUnitTests PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-G ${ARCH_FLAGS}>) # always with -MG
#endif()

set(CMAKE_VERBOSE_MAKEFILE OFF) # turn on if we want to debug in compilation.
