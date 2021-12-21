// #include "mlperf_sut.h"
#include "date.h"
using namespace date;

#include <grpcpp/grpcpp.h>

#ifdef BAZEL_BUILD
#include "examples/protos/helloworld.grpc.pb.h"
#else
#include "helloworld.grpc.pb.h"
#endif


using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReaderWriter;
using grpc::Status;
using helloworld::InferenceService;
using helloworld::InferenceService2;
using helloworld::InferenceResponse;
using helloworld::InferenceRequest;
using helloworld::endMessageResponse;
using helloworld::endMessageRequest;

namespace grpc_tests {

class InferenceServiceClient {
 public:
  InferenceServiceClient(std::shared_ptr<Channel> channel)
      : stub_(InferenceService::NewStub(channel)) {}


  void start(){
    std::cout << "start" << std::endl;
    stream = stub_->Inference(&context);
  }

  std::vector<float> runOne(std::vector<float> &input_data, std::int64_t* shape, int64_t rank){

    InferenceRequest request;
    request.mutable_data()->Add(input_data.begin(), input_data.end());
    request.mutable_shape()->Add(shape, shape+rank);

    // auto start = std::chrono::high_resolution_clock::now();
    stream->Write(request);
    InferenceResponse response;
    stream->Read(&response);
    // double d = std::chrono::duration<double, std::nano>(std::chrono::high_resolution_clock::now()-start).count();

    std::vector<float> results(response.data().begin(), response.data().end());
    return results;

  }

  void end(){
    stream->WritesDone();
    Status status = stream->Finish();
    if (!status.ok()) {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
    }
  }

 private:
  std::unique_ptr<InferenceService::Stub> stub_;
  std::unique_ptr< ::grpc::ClientReaderWriter< ::helloworld::InferenceRequest, ::helloworld::InferenceResponse>> stream;
  ClientContext context; 
};




class InferenceService2Client {
 public:
  InferenceService2Client(std::shared_ptr<Channel> channel)
      : stub_(InferenceService2::NewStub(channel)) {}

  std::vector<float> runOne(std::vector<float> &input_data, std::int64_t* shape, int64_t rank){
    InferenceRequest request;
    request.mutable_data()->Add(input_data.begin(), input_data.end());
    request.mutable_shape()->Add(shape, shape+rank);
    ClientContext context;
    InferenceResponse response;
    // auto start = std::chrono::high_resolution_clock::now();
    Status status = stub_->Inference2(&context, request, &response);
    // double d = std::chrono::duration<double, std::nano>(std::chrono::high_resolution_clock::now()-start).count();
    // duration += d;
    // count++;

    std::vector<float> results(response.data().begin(), response.data().end());
    return results;
  }

  void end(){

    endMessageRequest request;
    ClientContext context;
    endMessageResponse response;
    Status status = stub_->end(&context, request, &response);

    if (!status.ok()) {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
    }else{
      std::cout <<"end " << response.data()<< std::endl;
    }

    // std::cout <<"end1 "<< duration/count << std::endl;

    
  }

 private:
  std::unique_ptr<InferenceService2::Stub> stub_;
  int64_t count = 0;
  double duration = 0.0;
};

}