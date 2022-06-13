cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -W -Wall -pthread")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(AIU_grpc_test)

# find_library(_REFLECTION
#   NAMES grpc++_reflection
#   PATHS ${ONNX_COMPILER_BUILD_DIR}/lib) 

# find_library(_GRPC_GRPCPP
#   NAMES grpc++
#   PATHS ${ONNX_COMPILER_BUILD_DIR}/lib) 

# find_library(_GRPC_GRPCPP
#   NAMES grpc++
#   PATHS ${ONNX_COMPILER_BUILD_DIR}/lib) 

# set(_PROTOBUF_LIBPROTOBUF libprotobuf)
# set(_REFLECTION grpc++_reflection)
# set(_GRPC_GRPCPP grpc++)

get_filename_component(utils "../utils" ABSOLUTE)
get_filename_component(utils_path "${utils}" PATH)

message(STATUS "hellohello_${utils_path}")

add_executable(base example.cc)
target_link_libraries(base 
    gtest_main 
    gtest              
    hw_grpc_proto
    ${_REFLECTION}
    ${_GRPC_GRPCPP}
    ${_PROTOBUF_LIBPROTOBUF})
target_include_directories(base PRIVATE ${utils_path})
add_test(NAME base_test COMMAND base)