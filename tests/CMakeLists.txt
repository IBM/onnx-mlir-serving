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
