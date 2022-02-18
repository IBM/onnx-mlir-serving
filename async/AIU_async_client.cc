#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <fstream>
#include <vector>

#include <grpc/support/log.h>
#include <grpcpp/grpcpp.h>

#ifdef BAZEL_BUILD
#include "examples/protos/inference.grpc.pb.h"
#else
#include "inference.grpc.pb.h"
#endif

using grpc::Channel;
using grpc::ClientAsyncResponseReader;
using grpc::ClientContext;
using grpc::CompletionQueue;
using grpc::Status;
using inference::InferenceService;
using inference::InferenceResponse;
using inference::InferenceRequest;

class InferenceClient {
 public:
  explicit InferenceClient(std::shared_ptr<Channel> channel)
      : stub_(InferenceService::NewStub(channel)) {}

  // Assembles the client's payload and sends it to the server.
  void Inference(std::vector<float> &input_data, int64_t* shape, int64_t rank) {
    // Data we are sending to the server.
    InferenceRequest request;
    request.mutable_data()->Add(input_data.begin(), input_data.end());
    request.mutable_shape()->Add(shape, shape+rank);

    // Call object to store rpc data
    AsyncClientCall* call = new AsyncClientCall;

    // stub_->PrepareAsyncInference() creates an RPC object, returning
    // an instance to store in "call" but does not actually start the RPC
    // Because we are using the asynchronous API, we need to hold on to
    // the "call" instance in order to get updates on the ongoing RPC.
    call->response_reader =
        stub_->PrepareAsyncInference(&call->context, request, &cq_);

    // StartCall initiates the RPC call
    call->response_reader->StartCall();

    // Request that, upon completion of the RPC, "reply" be updated with the
    // server's response; "status" with the indication of whether the operation
    // was successful. Tag the request with the memory address of the call
    // object.
    call->response_reader->Finish(&call->reply, &call->status, (void*)call);
  }

  // Loop while listening for completed responses.
  // Prints out the response from the server.
  void AsyncCompleteRpc() {
    void* got_tag;
    bool ok = false;

    // Block until the next result is available in the completion queue "cq".
    while (cq_.Next(&got_tag, &ok)) {
      // The tag in this example is the memory location of the call object
      AsyncClientCall* call = static_cast<AsyncClientCall*>(got_tag);

      // Verify that the request was completed successfully. Note that "ok"
      // corresponds solely to the request for updates introduced by Finish().
      GPR_ASSERT(ok);

      if (call->status.ok()){
        std::cout << "Greeter received: " << std::endl;
        std::cout << call->reply.data().end() - call->reply.data().begin() << std::endl;
      }
      else
        std::cout << "RPC failed" << std::endl;

      // Once we're complete, deallocate the call object.
      delete call;
    }
  }

 private:
  // struct for keeping state and data information
  struct AsyncClientCall {
    // Container for the data we expect from the server.
    InferenceResponse reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // Storage for the status of the RPC upon completion.
    Status status;

    std::unique_ptr<ClientAsyncResponseReader<InferenceResponse>> response_reader;
  };

  // Out of the passed in Channel comes the stub, stored here, our view of the
  // server's exposed services.
  std::unique_ptr<InferenceService::Stub> stub_;

  // The producer-consumer queue we use to communicate asynchronously with the
  // gRPC runtime.
  CompletionQueue cq_;
};

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
      // if(BIGENDIAN)
      std::reverse(buffer, buffer + 4);
      tmp = (*(float*)buffer);
      imageData.push_back(tmp);
    }

    fp.close();
}

int main(int argc, char** argv) {
  // Instantiate the client. It requires a channel, out of which the actual RPCs
  // are created. This channel models a connection to an endpoint (in this case,
  // localhost at port 50051). We indicate that the channel isn't authenticated
  // (use of InsecureChannelCredentials()).
  InferenceClient client(grpc::CreateChannel(
      "localhost:50051", grpc::InsecureChannelCredentials()));

  std:: cout << argc << std::endl;
  std::vector<float> imageData;
  //image path as input argv[1], read image to float array
  readImage(argv[1], imageData);
  int64_t shape[3] = {7,1,204};
  int64_t rank = 3;

  // Spawn reader thread that loops indefinitely
  std::thread thread_ = std::thread(&InferenceClient::AsyncCompleteRpc, &client);

  for (int i = 0; i < 10; i++) {
    std::string user("world " + std::to_string(i));
    client.Inference(imageData, shape, rank);  // The actual RPC call!
  }

  std::cout << "Press control-c to quit" << std::endl << std::endl;
  thread_.join();  // blocks forever

  return 0;
}
