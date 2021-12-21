/*
 *
 * Copyright 2015 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include "date.h"
using namespace date;

#ifdef BAZEL_BUILD
#include "examples/protos/helloworld.grpc.pb.h"
#else
#include "helloworld.grpc.pb.h"
#endif

#include "OnnxMlirRuntime.h"
extern "C"{
OMTensorList *run_main_graph(OMTensorList *);
}

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReaderWriter;
using grpc::Status;
using helloworld::InferenceService;
using helloworld::InferenceResponse;
using helloworld::InferenceRequest;

// Logic and data behind the server's behavior.
class InferenceServiceServiceImpl final : public InferenceService::Service {
  Status Inference(ServerContext* context, 
                   ServerReaderWriter<InferenceResponse, InferenceRequest>* stream) override {

    InferenceRequest request;
    while (stream->Read(&request)) {
      auto start = std::chrono::high_resolution_clock::now();
      InferenceResponse response;
      runsample(request.data(), request.shape(), response.mutable_data());
      stream->Write(response);
      double d = std::chrono::duration<double, std::nano>(std::chrono::high_resolution_clock::now()-start).count();
      duration += d;
      count++;
      // if(count > stopCount){
      //   std::cout << duration/count << std::endl;
      //   count = 0;
      //   duration = 0.0;
      // }  
    }
    // runsample(request->data(), request->shape(), reply->mutable_data());
    // std::cout << request->shape()[0] << std::endl;
    std::cout << duration/count << std::endl;
    return Status::OK;
  }

  void runsample(google::protobuf::RepeatedField<float> imageData, google::protobuf::RepeatedField< google::protobuf::int64 > inputShape, google::protobuf::RepeatedField<float>* result){
    
    int64_t rank = inputShape.size();
    float *x1Data = imageData.mutable_data();
    int64_t *shape = inputShape.mutable_data();

    OMTensor *x1 = omTensorCreate(x1Data, shape, rank, ONNX_TYPE_FLOAT);
    OMTensor *list[1] = {x1};
    OMTensorList *input = omTensorListCreate(list,1);
    OMTensorList *outputList = run_main_graph(input);
    

    OMTensor *y = omTensorListGetOmtByIndex(outputList,0);
    float *prediction = (float *)omTensorGetDataPtr(y);
    int64_t *output_shape = omTensorGetShape(y);
    int resultsize = 1;
    for(int i = 0; i < omTensorGetRank(y); i++){
      resultsize *= output_shape[i];
    }
    // std::cout <<  std::chrono::system_clock::now() << " server send 1"  << std::endl;
    result->Add(prediction, prediction + resultsize);
  }

  int64_t stopCount = 10000;
  int64_t count = 0;
  double duration = 0.0;
  public:
    InferenceServiceServiceImpl(int count):stopCount(count){}
};

void RunServer(int count) {
  std::string server_address("10.10.170.2:50051");
  InferenceServiceServiceImpl service(count);

  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  ServerBuilder builder;
  builder.SetMaxReceiveMessageSize(16777216);
  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(&service);
  // Finally assemble the server.
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;

  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
}

int main(int argc, char** argv) {
  RunServer(atoi(argv[1]));

  return 0;
}
