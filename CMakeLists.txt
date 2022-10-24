
cmake_minimum_required(VERSION 3.14)
set(CMAKE_DEBUG_POSTFIX d)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -pthread")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


project(onnx-mlir-grpc-serving)

# cmake -DCMAKE_BUILD_TYPE=Release \
#       -DGRPC_DIR:STRING=/aivol/grpc \
#       -DONNX_COMPILER_DIR:STRING=/aivol/mlperf/AIU_bk/onnx-mlir-build  \
#       -DCMAKE_PREFIX_PATH=/aivol/grpc_install ..


# cmake -DCMAKE_BUILD_TYPE=Release -DONNX_COMPILER_DIR:STRING=/home/ubuntu/onnx-mlir -DCMAKE_PREFIX_PATH=/aivol/grpc_install ..
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# if(DEFINED GRPC_DIR)
#   message(STATUS "GPRC_DIR: ${GRPC_DIR}")
#   # add_subdirectory(${GRPC_DIR} ${CMAKE_CURRENT_BINARY_DIR}/grpc EXCLUDE_FROM_ALL)
#   # set(_PROTOBUF_LIBPROTOBUF libprotobuf)
#   # set(_REFLECTION grpc++_reflection)
#   # set(_GRPC_GRPCPP grpc++)
# else()
#   find_package(Protobuf CONFIG REQUIRED)
#   include_directories(${Protobuf_INCLUDE_DIR})
#   find_package(gRPC CONFIG REQUIRED)
#   set(_REFLECTION gRPC::grpc++_reflection)
#   set(_GRPC_GRPCPP gRPC::grpc++)
# endif()
# if(DEFINED ONNX_COMPILER_DIR)
#   message(STATUS "ONNX_COMPILER_DIR: ${ONNX_COMPILER_DIR}")
# endif()


message("-- CMAKE_HOST_SYSTEM_PROCESSOR: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "s390x" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBIGENDIAN=1")
ELSE()
 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBIGENDIAN=0")
ENDIF()



# find_program(_PROTOBUF_PROTOC protoc)

# find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)

# message(STATUS "_PROTOBUF_PROTOC: ${_PROTOBUF_PROTOC}")
# message(STATUS "_GRPC_CPP_PLUGIN_EXECUTABLE: ${_GRPC_CPP_PLUGIN_EXECUTABLE}")
# # Proto file
# get_filename_component(onnx_proto "./utils/onnx.proto" ABSOLUTE)
# get_filename_component(hw_proto "./utils/inference.proto" ABSOLUTE)
# get_filename_component(hw_proto_path "${hw_proto}" PATH)

# # Generated sources
# set(onnx_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.cc")
# set(onnx_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.h")
# set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/inference.pb.cc")
# set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/inference.pb.h")
# set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/inference.grpc.pb.cc")
# set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/inference.grpc.pb.h")
# add_custom_command(
#       OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}" "${onnx_proto_srcs}" "${onnx_proto_hdrs}"
#       COMMAND ${_PROTOBUF_PROTOC}
#       ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
#         --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
#         -I "${hw_proto_path}"
#         --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
#         "${hw_proto}" "${onnx_proto}" 
#       DEPENDS "${hw_proto}")

# Include generated *.pb.h files


# hw_grpc_proto
# add_library(hw_grpc_proto ${onnx_proto_srcs} ${onnx_proto_hdrs} ${hw_grpc_srcs} ${hw_grpc_hdrs} ${hw_proto_srcs} ${hw_proto_hdrs})
# target_link_libraries(hw_grpc_proto
#   ${_REFLECTION}
#   ${_GRPC_GRPCPP}
#   )
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

