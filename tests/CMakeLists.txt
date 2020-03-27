find_package(GTest QUIET)

if(NOT GTEST_FOUND)
  message(STATUS "GTest not found. Tests disabled.")
  return()
endif()

enable_testing()


function(AD_TEST NAME)
  set(TEST_TARGET test_${NAME})
  add_executable(${TEST_TARGET} ${NAME}_test.cc)
  message(STATUS "Test enabled:      ${NAME}")


  target_link_libraries(${TEST_TARGET} PUBLIC ${LIB_NAME} GTest::GTest GTest::Main)
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
