cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -W -Wall -pthread")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(grpc_cpp_client)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(Benchmark "${CMAKE_CURRENT_SOURCE_DIR}/benchmark.cc")
target_link_libraries(Benchmark 
              hw_grpc_proto
              ${_REFLECTION}
              ${_GRPC_GRPCPP}
              )

add_executable(Client "${CMAKE_CURRENT_SOURCE_DIR}/main.cc")
target_link_libraries(Client 
              hw_grpc_proto
              ${_REFLECTION}
              ${_GRPC_GRPCPP}
              )

