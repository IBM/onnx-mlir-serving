
cmake_minimum_required(VERSION 3.14)
set(CMAKE_DEBUG_POSTFIX d)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -W -Wall -pthread")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(onnx-mlir-grpc-serving)


if(DEFINED GRPC_DIR)
  message(STATUS "ONNX_COMPILER: ${ONNX_COMPILER}")
else()
  message(FATAL_ERROR "Provide ONNX_COMPILER pls")
endif()
if(DEFINED GRPC_DIR)
  message(STATUS "GPRC_DIR: ${GRPC_DIR}")
else()
  message(FATAL_ERROR "Provide GPRC_DIR pls")
endif()
if(DEFINED ONNX_COMPILER_DIR)
  message(STATUS "ONNX_COMPILER_DIR: ${ONNX_COMPILER_DIR}")
else()
  message(FATAL_ERROR "Provide ONNX_COMPILER_DIR pls:${ONNX_COMPILER_DIR}")
endif()


message("-- CMAKE_HOST_SYSTEM_PROCESSOR: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "s390x" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBIGENDIAN=1")
ELSE()
 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBIGENDIAN=0")
ENDIF()

# add_subdirectory(/aivol/grpc/ ${CMAKE_CURRENT_BINARY_DIR}/grpc EXCLUDE_FROM_ALL)
add_subdirectory(${GRPC_DIR} ${CMAKE_CURRENT_BINARY_DIR}/grpc EXCLUDE_FROM_ALL)
set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_REFLECTION grpc++_reflection)
set(_GRPC_GRPCPP grpc++)


find_program(_PROTOBUF_PROTOC protoc)

find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)

message(STATUS "_PROTOBUF_PROTOC: ${_PROTOBUF_PROTOC}")
message(STATUS "_GRPC_CPP_PLUGIN_EXECUTABLE: ${_GRPC_CPP_PLUGIN_EXECUTABLE}")
# Proto file
get_filename_component(hw_proto "./utils/inference.proto" ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)

# Generated sources
set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/inference.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/inference.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/inference.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/inference.grpc.pb.h")
add_custom_command(
      OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${hw_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${hw_proto}"
      DEPENDS "${hw_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# hw_grpc_proto
add_library(hw_grpc_proto
  ${hw_grpc_srcs}
  ${hw_grpc_hdrs}
  ${hw_proto_srcs}
  ${hw_proto_hdrs})
target_link_libraries(hw_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
  )

# For onnx-mlir
find_library(CRuntime
  NAMES cruntime
  PATHS ${ONNX_COMPILER_DIR}/lib) 
message(STATUS "CRuntime: ${CRuntime}")



add_executable(Benchmark "grpc_client/benchmark.cc")
target_link_libraries(Benchmark 
              hw_grpc_proto
              ${_REFLECTION}
              ${_GRPC_GRPCPP}
              ${_PROTOBUF_LIBPROTOBUF})
target_include_directories(Benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(Client "grpc_client/main.cc")
target_link_libraries(Client 
              hw_grpc_proto
              ${_REFLECTION}
              ${_GRPC_GRPCPP}
              ${_PROTOBUF_LIBPROTOBUF})
target_include_directories(Client PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(grpc_server "grpc_server/grpc_server.cc" "grpc_server/thread_pool.cc" "grpc_server/model_loader.cc")        
target_link_libraries(grpc_server
              ${CRuntime}
              hw_grpc_proto
              ${_REFLECTION}
              ${_GRPC_GRPCPP}
              ${_PROTOBUF_LIBPROTOBUF})
target_include_directories(grpc_server PRIVATE
              ${ONNX_COMPILER_DIR}/include)

add_subdirectory(tests ${CMAKE_CURRENT_BINARY_DIR}/tests EXCLUDE_FROM_ALL)
