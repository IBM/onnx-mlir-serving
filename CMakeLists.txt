
cmake_minimum_required(VERSION 3.14)
set(CMAKE_DEBUG_POSTFIX d)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -pthread")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


project(onnx-mlir-grpc-serving)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

message("-- CMAKE_HOST_SYSTEM_PROCESSOR: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "s390x" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBIGENDIAN=1")
ELSE()
 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBIGENDIAN=0")
ENDIF()


add_subdirectory(utils ${CMAKE_CURRENT_BINARY_DIR}/utils)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
# For onnx-mlir
find_library(CRuntime
  NAMES cruntime
  PATHS ${ONNX_COMPILER_DIR}/lib) 
message(STATUS "CRuntime: ${CRuntime}")

include_directories("${CMAKE_CURRENT_BINARY_DIR}")

add_executable(grpc_server "grpc_server/grpc_server.cc" "grpc_server/thread_pool.cc" "grpc_server/model_loader.cc")        
target_link_libraries(grpc_server
              ${CRuntime}
              hw_grpc_proto
              ${_REFLECTION}
              ${_GRPC_GRPCPP}
              )
target_include_directories(grpc_server PRIVATE
              ${ONNX_COMPILER_DIR}/include)

add_subdirectory(tests ${CMAKE_CURRENT_BINARY_DIR}/tests EXCLUDE_FROM_ALL)
add_subdirectory(example_client/cpp ${CMAKE_CURRENT_BINARY_DIR}/cpp)
# add_subdirectory(example_client/python ${CMAKE_CURRENT_BINARY_DIR}/python)
