cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -W -Wall -pthread")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(grpc_server_test)


get_filename_component(utils "../utils" ABSOLUTE)
get_filename_component(utils_path "${utils}" PATH)

message(STATUS "hellohello_${utils_path}")

add_executable(grpc-test grpc-test.cc)
target_link_libraries(grpc-test 
    gtest_main 
    gtest              
    hw_grpc_proto
    ${_REFLECTION}
    ${_GRPC_GRPCPP}
    ${_PROTOBUF_LIBPROTOBUF})
target_include_directories(grpc-test PRIVATE ${utils_path})
add_test(NAME grpc-test COMMAND grpc-test)
