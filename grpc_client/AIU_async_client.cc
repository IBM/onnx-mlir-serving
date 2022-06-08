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
#include <thread>
#include <vector>

#include <grpc/support/log.h>
#include <grpcpp/grpcpp.h>

#include "inference.grpc.pb.h"

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

    std::ifstream fp2(imagePath +"/config.txt");
    if (!fp2.is_open()) {
      printf("read model config error\n");
      return -1;
    }
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

  private:
    std::string imagePath;
    std::vector<std::string> imageList;
    std::vector<int> imageLabels;
    std::map<long, std::vector<float>> imageInMemory;
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

  // void Inference(std::vector<float> &input_data, int64_t* shape, int64_t rank) {
  Timer Inference(Dataset *ds, size_t index){

    auto input_data = ds->getImageData(index);

    InferenceRequest request;
    request.mutable_data()->Add(input_data.begin(), input_data.end());
    request.mutable_shape()->Add(ds->shape, ds->shape+ds->rank);
    request.set_model_name("ccf1");

    InferenceResponse reply;
    ClientContext context;
    CompletionQueue cq;
    Status status;
    std::unique_ptr<ClientAsyncResponseReader<InferenceResponse>> response_reader(stub_->PrepareAsyncInference(&context, request, &cq));

    Timer ir;
    ir.start = high_resolution_clock::now();
    response_reader->StartCall();
    response_reader->Finish(&reply, &status, (void*)1);
    void* got_tag;
    bool ok = false;

    GPR_ASSERT(cq.Next(&got_tag, &ok));
    GPR_ASSERT(got_tag == (void*)1);
    GPR_ASSERT(ok);

    ir.end = high_resolution_clock::now();
   
    // if (status.ok()) {
    //   std::cout << "ok" << std::endl;
    // } else {
    //   std::cout << "failed" << std::endl;
    // }
    return ir;
  }

 private:
  std::unique_ptr<InferenceService::Stub> stub_;
};

class Simlate{

  double _recordStart;
  double _recordEnd; 
  double _totalTime;
  std::vector<std::thread> _client_threads;
  std::mutex _log_mutex;
  Dataset &ds;
  char _host[200];
  std::ofstream ofs;

  public:
    Simlate(char* host, Dataset &ds, int threadNum, double recordStart, double recordEnd, double totalTime, char* logPrefix):ds(ds){

      _recordStart = recordStart*1000;
      _recordEnd = recordEnd*1000; 
      _totalTime = totalTime*1000;

      // char* _host = host;
      memcpy(_host, host, strlen(host));

      char logname[200]; 
      sprintf(logname, "build/log_%s.txt", logPrefix);

      ofs.open(logname, std::ios::out);

      createThread(threadNum);
      for (std::thread& thread : _client_threads) {
          //thread.detach();
          if(thread.joinable())
              thread.join();
      }

      ofs.close();

    }

    void createThread(int threadNum){
      high_resolution_clock::time_point startTime = high_resolution_clock::now();
      for(size_t i = 0; i<threadNum; i++){
        
        _client_threads.emplace_back([this](high_resolution_clock::time_point startTime, size_t threadIndex){
          InferenceClient client(grpc::CreateChannel(_host, grpc::InsecureChannelCredentials()));

          // std::cout << "Using " << _host << std::endl;

          size_t imageSize = ds.getImageCount();
          //ds.shape, ds.rank
          int count = 0;
          bool isStartRecord = false;
          bool isEndRecord = false;
          bool isRun = false;
          int totalcount = 0;
          std::vector<Timer> timerRecord;
          while(true){
            int index = count % imageSize;
            // std::vector<float> data = ds.getImageData(index);
            if(!isStartRecord){
              double d = std::chrono::duration<double, std::milli>(high_resolution_clock::now() - startTime).count();
              // std::cout << d << " " << _recordStart << std::endl;
              if(d > _recordStart){
                isStartRecord = true;
                // std::cout << d << " " << _recordStart << " start record" << std::endl;
              }
                
            }else{
              double d = std::chrono::duration<double, std::milli>(high_resolution_clock::now() - startTime).count();
              if(d > _recordEnd){
                isEndRecord = true;
                // std::cout << d << " " << _recordEnd<< " end record" << std::endl;
              }
                
            }
              
            if(!isRun){
              double d = std::chrono::duration<double, std::milli>(high_resolution_clock::now() - startTime).count();
              if(d > 5000){
                isRun = true;
                // std::cout << d << " " << _recordEnd<< " end record" << std::endl;
              }else{
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
              }
            }

            if(isRun){
              Timer r = client.Inference(&ds, index);
              if(isStartRecord && !isEndRecord){
                timerRecord.push_back(r);
              }
              count ++;
              totalcount ++;
            }

              
            if(isEndRecord){
              double d = std::chrono::duration<double, std::milli>(high_resolution_clock::now() - startTime).count();
              if( d > _totalTime ){
                for(Timer t:timerRecord){
                  std::lock_guard<std::mutex> log_lock(_log_mutex);
                  // std::cout << threadIndex << " ";
                  // std::cout << std::to_string(std::chrono::duration<double, std::nano>(t.end -t.start).count()) << std::endl;
                  ofs << threadIndex << " ";
                  ofs << std::to_string(std::chrono::duration<double, std::nano>(t.end -t.start).count()) << std::endl;
                }
                break;
              }
            }

            
          }
        },startTime, i);
      }
    }
};

// ./AIU_async_client /aivol/inputs/ccf1_inputs 1 1
int main(int argc, char** argv) {

  Dataset ds(argv[1]);
  int threadNum = 64;
  char* logPrefix = "out1";
  if(argc > 2){
    logPrefix = argv[2];
  }

  if(argc > 3){
    threadNum = std::stoi(argv[3]);
  }



  char* host = "localhost:50051";
  if((host = getenv("AIU_server"))){
    std::cout << "Using " << host << std::endl;
  }else{
    host = "localhost:50051";
    std::cout << "Using " << host << std::endl;
  }

  std::cout << threadNum << std::endl;  
  Simlate s(host, ds, threadNum, 20, 280, 300, logPrefix);



  InferenceClient* client = new InferenceClient(grpc::CreateChannel( host, grpc::InsecureChannelCredentials()));
  client->printStatistics();


  return 0;
}
