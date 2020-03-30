cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(LIB_NAME AutodiffCodegen)
project (${LIB_NAME} VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH} )


if (NOT CMAKE_BUILD_TYPE)
  message("-- No build type specified; defaulting to CMAKE_BUILD_TYPE=Release.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
endif()

#eigen
find_package(Eigen3 REQUIRED)


find_package(Glog REQUIRED)
  
add_subdirectory(src)




add_subdirectory(tests)
add_subdirectory(benchmarks)






