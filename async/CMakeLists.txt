
cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -W -Wall -pthread")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(AIU_grpc)

# include(../cmake/common.cmake)
# Provide grpc installation directory
if(DEFINED GRPC_DIR)
  message(STATUS "GPRC_DIR: ${GRPC_DIR}")
else()
  message(FATAL_ERROR "Provide GPRC_DIR pls:${GRPC_DIR}")
endif()
if(DEFINED ONNX_COMPILER_BUILD_DIR)
  message(STATUS "ONNX_COMPILER_BUILD_DIR: ${ONNX_COMPILER_BUILD_DIR}")
else()
  message(FATAL_ERROR "Provide ONNX_COMPILER_BUILD_DIR pls:${ONNX_COMPILER_BUILD_DIR}")
endif()
if(DEFINED LOADGEN_DIR)
  message(STATUS "LOADGEN_DIR: ${LOADGEN_DIR}")
else()
  message(FATAL_ERROR "Provide LOADGEN_DIR pls:${LOADGEN_DIR}")
endif()

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
get_filename_component(hw_proto "./inference.proto" ABSOLUTE)
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

# For dlc
find_library(CRuntime
  NAMES cruntime
  PATHS ${ONNX_COMPILER_BUILD_DIR}/lib) 

message(STATUS "CRuntime: ${CRuntime}")


# configure_file("./library.so"
#   ${CMAKE_CURRENT_BINARY_DIR}/library.so COPYONLY)

# add_library(OMRuntimeTestModel SHARED IMPORTED)
# set_property(TARGET OMRuntimeTestModel
#             PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/library.so)

# for loadgen
set(LOADGEN_LIB_DIR ${LOADGEN_DIR}/build)
find_library(mlperf_loadgen
	NAMES mlperf_loadgen
	PATHS ${LOADGEN_LIB_DIR})

# Targets greeter_[async_](client|server)
foreach(_target
  AIU_async_client)
  add_executable(${_target} "${_target}.cc")
  target_link_libraries(${_target}
    hw_grpc_proto
    ${_REFLECTION}
    ${_GRPC_GRPCPP}
    ${_PROTOBUF_LIBPROTOBUF})
endforeach()

foreach(_target
  AIU_async_server)
  add_executable(${_target} "${_target}.cc")
  target_link_libraries(${_target}
    # ${zdnn}
    # ${CMAKE_CURRENT_BINARY_DIR}/library.so
    ${CRuntime}
    hw_grpc_proto
    ${_REFLECTION}
    ${_GRPC_GRPCPP}
    ${_PROTOBUF_LIBPROTOBUF})
  target_include_directories(${_target} PRIVATE
    ${ONNX_COMPILER_BUILD_DIR}/include)
endforeach()


foreach(_target
  sut single multip)
  add_executable(${_target} "${_target}.cc")
  target_link_libraries(${_target}
    # ${zdnn}
    # ${CMAKE_CURRENT_BINARY_DIR}/library.so
    ${CRuntime}
    ${mlperf_loadgen}
    hw_grpc_proto
    ${_REFLECTION}
    ${_GRPC_GRPCPP}
    ${_PROTOBUF_LIBPROTOBUF})
  target_include_directories(${_target} PRIVATE
    ${ONNX_COMPILER_BUILD_DIR}/include
    ${LOADGEN_DIR})
endforeach()