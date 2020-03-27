

add_library(${LIB_NAME} SHARED)
target_include_directories(${LIB_NAME} PUBLIC ".")
target_link_libraries(${LIB_NAME} Eigen3::Eigen glog::glog)


#############  source #############

# Group source files in IDEs
FILE(GLOB_RECURSE ALL_SRC  *.cc)
FILE(GLOB_RECURSE ALL_HEADER  *.h)
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} FILES  ${ALL_SRC} ${ALL_HEADER})

# Source files in "src/" to level directory
FILE(GLOB_RECURSE PROG_SRC  *.cc)
FILE(GLOB_RECURSE PROG_HEADER  *.h ${SAIGA_BUILDCONFIG_FILE})
target_sources(${LIB_NAME}  PRIVATE ${PROG_SRC} PUBLIC ${PROG_HEADER})

