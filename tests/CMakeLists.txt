find_package(GTest QUIET)

if(NOT GTEST_FOUND)
  message(STATUS "GTest not found. Tests disabled.")
  return()
endif()


### Build Test Util Library

add_library(test_util STATIC test_utils.cc)
target_link_libraries(test_util ${LIB_NAME})


enable_testing()


function(AD_TEST NAME)
  set(TEST_TARGET test_${NAME})
  add_executable(${TEST_TARGET} ${NAME}_test.cc)
  message(STATUS "Test enabled:      ${NAME}")


  target_link_libraries(${TEST_TARGET} PUBLIC ${LIB_NAME} GTest::GTest GTest::Main test_util)
  target_link_libraries(${TEST_TARGET} PUBLIC ${ARGN})

  add_test(
    NAME ${TEST_TARGET}
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${PROG_NAME}
    )

  #set working directory for visual studio so the project can be executed from the ide
  set_target_properties(${PROG_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${OUTPUT_DIR}")
  set_target_properties(${PROG_NAME} PROPERTIES FOLDER tests/${PREFIX})
  set_target_properties(${PROG_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
endfunction()

AD_TEST(code_generator)
AD_TEST(expression_graph)
AD_TEST(expression_ref)
AD_TEST(expression)


macro (generate_test_functor FUNCTOR_NAME FUNCTOR_FILE)
  ceres_generate_cost_function_implementation_for_functor(
    NAME ${FUNCTOR_NAME}
    INPUT_FILE ${FUNCTOR_FILE}
    OUTPUT_DIRECTORY tests
    NAMESPACE test
    )
endmacro()


# Semantic tests should work on every platform
include(CeresCodeGeneration)


generate_test_functor(InputOutputAssignment autodiff_codegen_test.h)
generate_test_functor(CompileTimeConstants autodiff_codegen_test.h)
generate_test_functor(Assignments autodiff_codegen_test.h)
generate_test_functor(BinaryArithmetic autodiff_codegen_test.h)
generate_test_functor(UnaryArithmetic autodiff_codegen_test.h)
generate_test_functor(BinaryComparison autodiff_codegen_test.h)
generate_test_functor(LogicalOperators autodiff_codegen_test.h)
generate_test_functor(ScalarFunctions autodiff_codegen_test.h)
generate_test_functor(LogicalFunctions autodiff_codegen_test.h)
generate_test_functor(Branches autodiff_codegen_test.h)

AD_TEST(autodiff_codegen InputOutputAssignment CompileTimeConstants Assignments BinaryArithmetic
  UnaryArithmetic BinaryComparison LogicalOperators ScalarFunctions
  LogicalFunctions Branches)

#target_link_libraries(autodiff_codegen_test PUBLIC
#  )
