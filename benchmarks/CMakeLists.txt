find_package(benchmark QUIET)

if(NOT benchmark_FOUND)
  message(STATUS "google benchmark not found. Tests disabled.")
  return()
endif()

# TODO: Add support for other compilers
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  list(APPEND CERES_BENCHMARK_FLAGS "-mllvm" "-inline-threshold=1000000")
  list(APPEND CERES_BENCHMARK_FAST_MATH_FLAGS ${CERES_BENCHMARK_FLAGS} "-ffast-math")
endif()

macro(AD_BENCHMARK NAME)

  set(BM_TARGET ${NAME})
  add_executable(${BM_TARGET} ${NAME}.cc)
  target_link_libraries(${BM_TARGET} PUBLIC ${LIB_NAME} benchmark::benchmark)
  target_link_libraries(${BM_TARGET} PUBLIC ${ARGN})
  target_compile_options(${BM_TARGET} PRIVATE ${CERES_BENCHMARK_FLAGS})


  set(BM_TARGET_FAST_MATH ${NAME}_fast_math)
  add_executable(${BM_TARGET_FAST_MATH} ${NAME}.cc)
  target_link_libraries(${BM_TARGET_FAST_MATH} PUBLIC ${LIB_NAME} benchmark::benchmark)
  target_link_libraries(${BM_TARGET_FAST_MATH} PUBLIC ${ARGN})
  target_compile_options(${BM_TARGET_FAST_MATH} PRIVATE ${CERES_BENCHMARK_FAST_MATH_FLAGS})
endmacro()

include(CeresCodeGeneration)
macro (generate_benchmark_functor FUNCTOR_NAME FUNCTOR_FILE)
  ceres_generate_cost_function_implementation_for_functor(
    NAME ${FUNCTOR_NAME}
    INPUT_FILE ${FUNCTOR_FILE}
    OUTPUT_DIRECTORY benchmarks
    NAMESPACE ceres
    )
endmacro()

generate_benchmark_functor(SnavelyReprojectionError snavely_reprojection_error.h)
generate_benchmark_functor(Linear1CostFunction linear_cost_functions.h)
generate_benchmark_functor(Linear10CostFunction linear_cost_functions.h)
generate_benchmark_functor(Brdf brdf_cost_function.h)

AD_BENCHMARK(autodiff_benchmarks
  SnavelyReprojectionError Linear1CostFunction Linear10CostFunction Brdf)




