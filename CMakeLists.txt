
cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -W -Wall -pthread")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(HelloWorld C CXX)

# include(../cmake/common.cmake)
add_subdirectory(../../.. ${CMAKE_CURRENT_BINARY_DIR}/grpc EXCLUDE_FROM_ALL)
set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_REFLECTION grpc++_reflection)
set(_GRPC_GRPCPP grpc++)
# find_library(_REFLECTION NAMES grpc++_reflection /aivol/grpc/examples/cpp/helloworld/cmake/build/grpc)
# find_library(_GRPC_GRPCPP NAMES grpc++ /aivol/grpc/examples/cpp/helloworld/cmake/build/grpc)

find_program(_PROTOBUF_PROTOC protoc)
find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)

# Proto file
get_filename_component(hw_proto "./helloworld.proto" ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)

# Generated sources
set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.grpc.pb.h")
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
  ${_PROTOBUF_LIBPROTOBUF})

# For dlc
set(ONNX_MLIR_SRC_ROOT /aivol/DLCpp/third_party/onnx-mlir/)
find_library(CRuntime
  NAMES cruntime
  PATHS /aivol/DLCpp/dlc-build/Debug/lib)  

configure_file(/aivol/onnx-zoo-scripts/models_so/onnx-zoo/models/vision/classification/mnist/model/mnist-7.so
  ${CMAKE_CURRENT_BINARY_DIR}/library.so COPYONLY)

add_library(OMRuntimeTestModel SHARED IMPORTED)
set_property(TARGET OMRuntimeTestModel
            PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/library.so)

# for loadgen
set(LOADGEN_DIR /aivol/mlperf/inference1.0/loadgen)
set(LOADGEN_LIB_DIR ${LOADGEN_DIR}/build)
find_library(mlperf_loadgen
	NAMES mlperf_loadgen
	PATHS ${LOADGEN_LIB_DIR})

# Targets greeter_[async_](client|server)
foreach(_target
  greeter_client)
  add_executable(${_target} "${_target}.cc")
  target_link_libraries(${_target}
    hw_grpc_proto
    ${_REFLECTION}
    ${_GRPC_GRPCPP}
    ${_PROTOBUF_LIBPROTOBUF})
endforeach()

foreach(_target
  greeter_server greeter_server2 )
  add_executable(${_target} "${_target}.cc")
  target_link_libraries(${_target}
    ${CMAKE_CURRENT_BINARY_DIR}/library.so
    ${CRuntime}
    hw_grpc_proto
    ${_REFLECTION}
    ${_GRPC_GRPCPP}
    ${_PROTOBUF_LIBPROTOBUF})
  target_include_directories(${_target} PRIVATE
    ${ONNX_MLIR_SRC_ROOT}/include
    ${ONNX_MLIR_SRC_ROOT})
endforeach()

foreach(_target
  app )
  add_executable(${_target} "main.cc" "mlperf_sut.cc")
  target_link_libraries(${_target}
    ${CMAKE_CURRENT_BINARY_DIR}/library.so
    ${CRuntime}
    ${mlperf_loadgen}
    hw_grpc_proto
    ${_REFLECTION}
    ${_GRPC_GRPCPP}
    ${_PROTOBUF_LIBPROTOBUF})
  target_include_directories(${_target} PRIVATE
    ${ONNX_MLIR_SRC_ROOT}/include
    ${ONNX_MLIR_SRC_ROOT}
    ${LOADGEN_DIR})
endforeach()
