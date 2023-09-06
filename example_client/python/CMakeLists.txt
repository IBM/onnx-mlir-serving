project(grpc_python_client)


find_package(PythonInterp)

set(onnx_proto_grpc_py "${CMAKE_CURRENT_BINARY_DIR}/onnx.pb2_grpc.py")
set(onnx_proto_py "${CMAKE_CURRENT_BINARY_DIR}/onnx.pb2.py")
set(hw_proto_py "${CMAKE_CURRENT_BINARY_DIR}/inference.pb2.py")
set(hw_proto_grpc_py "${CMAKE_CURRENT_BINARY_DIR}/inference.pb2_grpc.py")


execute_process(COMMAND ${PYTHON_EXECUTABLE} -m grpc_tools.protoc --grpc_out "${CMAKE_CURRENT_BINARY_DIR}" --python_out= "${CMAKE_CURRENT_BINARY_DIR}" -I "${hw_proto_path}" "${hw_proto}" "${onnx_proto}" )


