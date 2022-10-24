#include <iostream>
#include <memory>
#include <string>
#include <fstream>
#include <thread>
#include <vector>
#include<numeric>


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
  Dataset(std::string imagePath_):imagePath(imagePath_){
    readImageList();
    for(size_t i = 0; i< getImageCount(); i++){
        loadImageData(i);
    }
  }
  void readImage(std::string imageName, std::vector<float> &imageData){
    std::string currentImagePath = imagePath + "/" + imageName;
    std::ifstream fp(currentImagePath, std::ios::in | std::ios::binary);

    if (!fp.is_open()) {
      std::cout << "read image "<< imageName << " error" << std::endl;
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
  int readImageList(){
    std::ifstream fp(imagePath +"/val_map.txt");

    if (!fp.is_open()) {
      std::cout << "read val_map error" << std::endl;
      return -1;
    }
    std::string imageName;
    int imageLabel;
    while(fp >> imageName >> imageLabel){
      imageList.push_back(imageName);
      imageLabels.push_back(imageLabel);
    }
    fp.close();

    std::ifstream fp2(imagePath +"/grpc_config.txt");
    if (!fp2.is_open()) {
      printf("read model config error\n");
      return -1;
    }
    char typeName[5];
    fp2 >> modelName;
    fp2 >> typeName >> dataTypeSize;
    fp2 >> rank;
    shape = (int64_t*)malloc(rank*sizeof(int64_t));
    int index = 0;
    while(index<rank){
      fp2 >> shape[index];
      index ++;
    }

    fp2.close();

    return 0;
  }
  void loadImageData(size_t index){
    std::vector<float> imageData;
    readImage(imageList[index], imageData);
    imageInMemory[index] = imageData;
  }
  int getImageLabel(int index){
    return imageLabels[index];
  }
  size_t getImageCount(){
    return imageList.size();
  }
  std::vector<float> getImageData(size_t index){
    return imageInMemory[index];
  }
  public:
    std::int64_t rank;
    std::int64_t* shape;
    std::string modelName;

  private:
    std::string imagePath;
    std::vector<std::string> imageList;
    std::vector<int> imageLabels;
    std::map<long, std::vector<float>> imageInMemory;
    int dataTypeSize;
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
    std::cout << "end done"<<std::endl;
  }      

  std::vector<float> Inference(std::vector<float> input_data, std::int64_t * shape, std::int64_t rank, std::string model_name){

    InferenceRequest request;

    onnx::TensorProto* tensor = request.add_tensor();

    tensor->set_data_type(1);
    tensor->mutable_dims()->Add(shape, shape+rank);
    tensor->mutable_float_data()->Add(input_data.data(), input_data.data() + input_data.size());
    request.set_model_name(model_name);
    
    // std::ofstream output(model_name + ".bin",std::ofstream::binary);
    // if(!request.SerializeToOstream(&output)){
    //   std::cout<<"save failed"<<std::endl;
    // }

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