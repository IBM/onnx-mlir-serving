project(onnx-parser)

if(DEFINED GRPC_DIR)
  message(STATUS "GPRC_DIR: ${GRPC_DIR}")
  add_subdirectory(${GRPC_DIR} ${CMAKE_CURRENT_BINARY_DIR}/grpc EXCLUDE_FROM_ALL)
  set(_PROTOBUF_LIBPROTOBUF libprotobuf)
  set(_REFLECTION grpc++_reflection)
  set(_GRPC_GRPCPP grpc++)
else()
  find_package(Protobuf CONFIG REQUIRED)
  include_directories(${Protobuf_INCLUDE_DIR})
  find_package(gRPC CONFIG REQUIRED)
  set(_REFLECTION gRPC::grpc++_reflection)
  set(_GRPC_GRPCPP gRPC::grpc++)
endif()
if(DEFINED ONNX_COMPILER_DIR)
  message(STATUS "ONNX_COMPILER_DIR: ${ONNX_COMPILER_DIR}")
endif()

find_program(_PROTOBUF_PROTOC protoc)
find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)

message(STATUS "_PROTOBUF_PROTOC: ${_PROTOBUF_PROTOC}")
message(STATUS "Protobuf_INCLUDE_DIR: ${Protobuf_INCLUDE_DIR}")
message(STATUS "_GRPC_CPP_PLUGIN_EXECUTABLE: ${_GRPC_CPP_PLUGIN_EXECUTABLE}")

get_filename_component(onnx_proto "${CMAKE_CURRENT_SOURCE_DIR}/onnx.proto" ABSOLUTE)
get_filename_component(hw_proto "${CMAKE_CURRENT_SOURCE_DIR}/inference.proto" ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)

# Generated sources
set(onnx_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.cc")
set(onnx_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.h")
set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/inference.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/inference.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/inference.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/inference.grpc.pb.h")
add_custom_command(
      OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}" "${onnx_proto_srcs}" "${onnx_proto_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${hw_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${hw_proto}" "${onnx_proto}" 
      DEPENDS "${hw_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# hw_grpc_proto
add_library(hw_grpc_proto ${onnx_proto_srcs} ${onnx_proto_hdrs} ${hw_grpc_srcs} ${hw_grpc_hdrs} ${hw_proto_srcs} ${hw_proto_hdrs})
target_link_libraries(hw_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  )

find_package(Protobuf REQUIRED)
message(STATUS "Protobuf: ${Protobuf_INCLUDE_DIR}")
include_directories(${Protobuf_INCLUDE_DIR})

add_executable(OnnxReader "onnx_reader.cc")
target_link_libraries(OnnxReader 
    hw_grpc_proto
    ${protobuf}
)