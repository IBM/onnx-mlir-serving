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
#include <fstream>

#include <grpcpp/grpcpp.h>

#ifdef BAZEL_BUILD
#include "examples/protos/helloworld.grpc.pb.h"
#else
#include "helloworld.grpc.pb.h"
#endif

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using helloworld::InferenceService;
using helloworld::InferenceResponse;
using helloworld::InferenceRequest;

void readImage(std::string currentImagePath, std::vector<float> &imageData)
{

    std::ifstream fp(currentImagePath, std::ios::in | std::ios::binary);

    if (!fp.is_open()) {
      printf("error\n");
      return;
    }

    float tmp;
    char buffer[4];
    while(fp.read(buffer, 4)){
      std::reverse(buffer, buffer + 4);
      tmp = (*(float*)buffer);
      imageData.push_back(tmp);
      // std::cout << tmp << std::endl;
    }

    fp.close();
}

class InferenceServiceClient {
 public:
  InferenceServiceClient(std::shared_ptr<Channel> channel)
      : stub_(InferenceService::NewStub(channel)) {}

  // Assembles the client's payload, sends it and presents the response back
  // from the server.
  float SayHello(std::string fileName) {
    // Data we are sending to the server.
    std::vector<float> imageData;
    readImage(fileName, imageData);

    ClientContext context; 
    auto stream = stub_->Inference(&context);

    InferenceRequest request;
    std::vector<int64_t> shape{ 1, 1, 28, 28};
    request.mutable_data()->Add(imageData.begin(), imageData.end());
    request.mutable_shape()->Add(shape.begin(), shape.end());

    stream->Write(request);

    // Container for the data we expect from the server.
    InferenceResponse response;

    stream->Read(&response);

    for(auto a: response.data()){
      std::cout << a << std::endl;
    }
    float result = response.data()[0];
    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    // ClientContext context;

    // The actual RPC.
    // Status status = stub_->Inference(&context, request, &reply);

    // Act upon its status.
    stream->WritesDone();
    Status status = stream->Finish();
    if (status.ok()) {
      return result;
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
      return -1.0;
    }
  }

 private:
  std::unique_ptr<InferenceService::Stub> stub_;
};

int main(int argc, char** argv) {
  // Instantiate the client. It requires a channel, out of which the actual RPCs
  // are created. This channel models a connection to an endpoint specified by
  // the argument "--target=" which is the only expected argument.
  // We indicate that the channel isn't authenticated (use of
  // InsecureChannelCredentials()).
  std::string target_str;
  // std::string arg_str("--target");
  // if (argc > 1) {
  //   std::string arg_val = argv[1];
  //   size_t start_pos = arg_val.find(arg_str);
  //   if (start_pos != std::string::npos) {
  //     start_pos += arg_str.size();
  //     if (arg_val[start_pos] == '=') {
  //       target_str = arg_val.substr(start_pos + 1);
  //     } else {
  //       std::cout << "The only correct argument syntax is --target="
  //                 << std::endl;
  //       return 0;
  //     }
  //   } else {
  //     std::cout << "The only acceptable argument is --target=" << std::endl;
  //     return 0;
  //   }
  // } else {
  //   target_str = "localhost:50051";
  // }
  target_str = "0.0.0.0:50051";
  InferenceServiceClient InferenceService(
      grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));

  float reply = InferenceService.SayHello(argv[1]);
  std::cout << "InferenceService received: " << reply << std::endl;


  return 0;
}
