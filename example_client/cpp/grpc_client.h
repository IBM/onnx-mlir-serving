#include <iostream>
#include <memory>
#include <string>
#include <fstream>
#include <thread>
#include <vector>
#include<numeric>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <regex>
#include <fcntl.h>
#include <unistd.h>


#include <grpc/support/log.h>
#include <grpcpp/grpcpp.h>

#include "utils/inference.grpc.pb.h"

using grpc::Channel;
using grpc::ClientAsyncResponseReader;
using grpc::ClientContext;
using grpc::CompletionQueue;
using grpc::Status;
using inference::InferenceService;
using inference::InferenceResponse;
using inference::InferenceRequest;
using inference::PrintStatisticsRequest;
using inference::PrintStatisticsResponse;
using std::chrono::high_resolution_clock;

struct Timer {
  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
};

class Dataset {
  public:
    std::vector<InferenceRequest*> requestsInput;
    std::string modelName;
    bool invalid = false;
    Dataset(std::string imagePath_, std::string modelName):
      modelName(modelName)
    {
      std::string imagePath = imagePath_;

      try{
          std::filesystem::path p = imagePath; 
          for (const auto &entry : std::filesystem::directory_iterator(p))
          {
              std::string path = entry.path().string();

              std::regex pattern(".*test_data_set_[0-9].*");

              if (std::regex_match(path, pattern)){
                  // std::cout << path << std::endl;
                  std::vector<std::string> paths;

                  for (const auto &inner_entry : std::filesystem::directory_iterator(entry.path())){
                      std::string inner_path = inner_entry.path().string();
                      if (inner_path.find("_input") == std::string::npos && inner_path.find("input") != std::string::npos)
                      {
                          // std::cout << inner_path << std::endl;
                          paths.push_back(inner_path);
                      }
                  }
                  std::sort(paths.begin(), paths.end());
                  InferenceRequest* request = new InferenceRequest();
                  request->set_model_name(modelName);
                  for (std::string p_: paths){
                      // std::cout << p << std::endl;
                      addTensor(p_.c_str(),request);
                  }
                  requestsInput.push_back(request);
              }
          }
      }catch (const std::filesystem::filesystem_error& ex) {
          std::cout << "Input directory cannot open directory: No such file or directory" <<std::endl;
          invalid = true;
      }
      if(requestsInput.size() == 0){
          std::cout << "No test_data_set_*/input_*.pb in input directory" <<std::endl;
          invalid = true;
      }
    }
    
  void addTensor(std::string imageName, InferenceRequest* request)
  {
    std::ifstream input(imageName, std::ios::ate | std::ios::binary);

    std::streamsize size = input.tellg(); // get current position in file
    input.seekg(0, std::ios::beg);        // move to start of file
    std::vector<char> buffer(size);
    input.read(buffer.data(), size); // read raw data


    onnx::TensorProto* tensor = request->add_tensor();
    tensor->ParseFromArray(buffer.data(), size);
    input.close();
  }

  InferenceRequest* getInput(int index){
    return requestsInput[index];
  }

  int getImageCount(){
    return requestsInput.size();
  }

};

class InferenceClient {
 public:
  explicit InferenceClient(std::shared_ptr<Channel> channel)
      : stub_(InferenceService::NewStub(channel)) {}

  void printStatistics(){
    PrintStatisticsRequest request;
    PrintStatisticsResponse response;
    ClientContext context;
    CompletionQueue cq;
    Status status;
    std::unique_ptr<ClientAsyncResponseReader<PrintStatisticsResponse> > rpc(
            stub_->PrepareAsyncPrintStatistics(&context, request, &cq));
    rpc->StartCall();
    rpc->Finish(&response, &status, (void*)1);
    void* got_tag;
    bool ok = false;
    GPR_ASSERT(cq.Next(&got_tag, &ok));
    GPR_ASSERT(got_tag == (void*)1);
  }      

  std::vector<float> Inference(InferenceRequest* request_){  
    
    InferenceRequest request = *request_;

    const onnx::TensorProto& tensor = request.tensor(0);

    InferenceResponse reply;
    ClientContext context;
    CompletionQueue cq;
    Status status;
    std::unique_ptr<ClientAsyncResponseReader<InferenceResponse>> response_reader(stub_->PrepareAsyncInference(&context, request, &cq));
    response_reader->StartCall();
    response_reader->Finish(&reply, &status, (void*)1);
    void* got_tag;
    bool ok = false;

    GPR_ASSERT(cq.Next(&got_tag, &ok));
    GPR_ASSERT(got_tag == (void*)1);
    GPR_ASSERT(ok);
    std::vector<float> out(reply.tensor(0).float_data().begin(),reply.tensor(0).float_data().end());
    return out;
  }

 private:
  std::unique_ptr<InferenceService::Stub> stub_;
};