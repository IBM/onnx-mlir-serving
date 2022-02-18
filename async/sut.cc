/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
 */
#include <cassert>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <fstream>
#include <map>
#include <mutex>
#include <thread>
#include <vector>
#include <string>
#include <sstream>
// #include <algorithm> 

#include <future>


#include "loadgen.h"
#include "query_sample_library.h"
#include "system_under_test.h"
#include "test_settings.h"

#include "OnnxMlirRuntime.h"

extern "C"{
OMTensorList *run_main_graph(OMTensorList *);
}

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
using inference::EndRequest;
using inference::EndResponse;


struct InferenceResult {
  mlperf::ResponseId id;
  float* result;
  int size;
};

class Dataset {
 public:
  Dataset(std::string imagePath_, bool grpcCall_):imagePath(imagePath_){
    readImageList();
    grpcCall = grpcCall_;
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
  void loadImageData(long index){
    std::vector<float> imageData;
    readImage(imageList[index], imageData);
    imageInMemory[index] = imageData;
  }
  int getImageLabel(int index){
    return imageLabels[index];
  }
  long getImageCount(){
    return imageList.size();
  }
  std::vector<float> getImageData(long index){
    return imageInMemory[index];
  }
  public:
    std::int64_t rank;
    std::int64_t* shape;
    bool grpcCall  = true;

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

  void end(){
    EndRequest request;
    EndResponse response;
    ClientContext context;
    CompletionQueue cq;
    Status status;
    std::unique_ptr<ClientAsyncResponseReader<EndResponse> > rpc(
            stub_->PrepareAsyncEnd(&context, request, &cq));
    rpc->StartCall();
    rpc->Finish(&response, &status, (void*)1);
  }

  std::vector<InferenceResult> runItems( Dataset &ds, const std::vector<mlperf::QuerySample>& samples ){
    // std::cout <<  "inference call " << std::endl;
    if(ds.grpcCall){
      std::vector<InferenceResult> results;
      for(auto sample: samples){
        auto input = ds.getImageData(sample.index);
        std::vector<float> output = Inference(input, ds.shape, ds.rank);
        results.push_back({sample.id,output.data(),output.size()});
      }
      return results;
    }else{
      return InferenceLocal(ds,samples);
    }
  }

  std::vector<float>  Inference(std::vector<float> &input_data, int64_t* shape, int64_t rank) {
    InferenceRequest request;
    request.mutable_data()->Add(input_data.begin(), input_data.end());
    request.mutable_shape()->Add(shape, shape+rank);
    InferenceResponse response;

    ClientContext context;
    CompletionQueue cq;
    Status status;
    std::unique_ptr<ClientAsyncResponseReader<InferenceResponse> > rpc(
            stub_->PrepareAsyncInference(&context, request, &cq));
    rpc->StartCall();
    rpc->Finish(&response, &status, (void*)1);
    void* got_tag;
    bool ok = false;
    GPR_ASSERT(cq.Next(&got_tag, &ok));
    GPR_ASSERT(got_tag == (void*)1);

    std::vector<float> results(response.data().begin(), response.data().end());
    return results;
  }

  std::vector<InferenceResult> InferenceLocal( Dataset &ds, const std::vector<mlperf::QuerySample>& samples ){

    std::vector<InferenceResult> results;
    for(auto s: samples){

    std::vector<float> f = ds.getImageData(s.index);

    int64_t* shape = ds.shape;
    int64_t rank = ds.rank;
    float *x1Data = f.data();
    OMTensor *x1 = omTensorCreate(x1Data, shape, rank, ONNX_TYPE_FLOAT);

    OMTensor *list[1] = {x1};
    OMTensorList *input = omTensorListCreate(list,1); //new OMTensorList(list,1);
    OMTensorList *outputList = run_main_graph(input);
    OMTensor *y = omTensorListGetOmtByIndex(outputList,0); // outputList->_omts[0];

    float *prediction = (float *)omTensorGetDataPtr(y);

    int64_t *output_shape = omTensorGetShape(y);

    int resultsize = 1;
    for(int i = 0; i < omTensorGetRank(y); i++){
      resultsize *= output_shape[i];
    }

      results.push_back({s.id,prediction,resultsize});
    }

    return results;
  }  

 private:
  std::unique_ptr<InferenceService::Stub> stub_;
};


std::vector<InferenceResult> runItems( Dataset &ds, const std::vector<mlperf::QuerySample>& samples ){

  std::vector<InferenceResult> results;
  for(auto s: samples){

   std::vector<float> f = ds.getImageData(s.index);

   int64_t* shape = ds.shape;
   int64_t rank = ds.rank;
   float *x1Data = f.data();
   OMTensor *x1 = omTensorCreate(x1Data, shape, rank, ONNX_TYPE_FLOAT);

   OMTensor *list[1] = {x1};
   OMTensorList *input = omTensorListCreate(list,1); //new OMTensorList(list,1);
   OMTensorList *outputList = run_main_graph(input);
   OMTensor *y = omTensorListGetOmtByIndex(outputList,0); // outputList->_omts[0];
   //float *prediction = (float *)y->_alignedPtr;
   float *prediction = (float *)omTensorGetDataPtr(y);
   // int resultsize = y->_shape[1];
   int64_t *output_shape = omTensorGetShape(y);

   int resultsize = 1;
   for(int i = 0; i < omTensorGetRank(y); i++){
     resultsize *= output_shape[i];
   }

    results.push_back({s.id,prediction,resultsize});
  }

  return results;
}

class QSL : public mlperf::QuerySampleLibrary {
 public:
  QSL(Dataset &ds):ds(ds){}
  ~QSL() = default;
  const std::string& Name() const override { return name_; }
  size_t TotalSampleCount() override { return ds.getImageCount(); }
  size_t PerformanceSampleCount() override { return 4; }

  void LoadSamplesToRam(
    const std::vector<mlperf::QuerySampleIndex>& samples) override {
    for(mlperf::QuerySampleIndex s: samples){
      ds.loadImageData(s);
    }
    return;
  }

  void UnloadSamplesFromRam(
      const std::vector<mlperf::QuerySampleIndex>& samples) override {
    return;
  }

 private:
  std::string name_{"QSL"};
  Dataset &ds;
};

class BasicSUT : public mlperf::SystemUnderTest {
 public:
  BasicSUT(Dataset &ds):ds(ds) {
    client = new InferenceClient(grpc::CreateChannel( "localhost:50051", grpc::InsecureChannelCredentials()));
  }
  ~BasicSUT() override {}
  const std::string& Name() const override { return mName; }
  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
    std::vector<mlperf::QuerySampleResponse> responses;
    responses.reserve(samples.size());
    auto results = client->runItems(ds,samples);

    for(auto result: results){
      mlperf::QuerySampleResponse res = {result.id, reinterpret_cast<std::uintptr_t>(result.result), result.size*sizeof(result.result[0])};
      responses.push_back(res);
    }
    mlperf::QuerySamplesComplete(responses.data(), responses.size());
  }
  void FlushQueries() override {}
  void ReportLatencyResults(
      const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override{};

 private:
  Dataset &ds;
  int mBuf{0};
  std::string mName{"BasicSUT"};
  InferenceClient* client;
};

class QueueSUT : public mlperf::SystemUnderTest {
 public:
  QueueSUT(int numCompleteThreads, Dataset &ds_):ds(ds_) {
    // Each thread handle at most maxSize at a time.

    // Launch complete threads
    for (int i = 0; i < numCompleteThreads; i++) {
      mThreads.emplace_back(&QueueSUT::CompleteThread, this, i);
    }
  }
  ~QueueSUT() override {
    {
      std::unique_lock<std::mutex> lck(mMtx);
      mDone = true;
      mCondVar.notify_all();
    }
    for (auto& thread : mThreads) {
      thread.join();
    }
  }
  const std::string& Name() const override { return mName; }
  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {

    std::unique_lock<std::mutex> lck(mMtx);
    for (const auto& sample : samples) {
      mIdQueue.push_back(sample);
    }

    mCondVar.notify_one();
  }
  void FlushQueries() override {}
  void ReportLatencyResults(
      const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override{};

 private:
  void CompleteThread(int threadIdx) {
    std::vector<mlperf::QuerySampleResponse> responses;
    size_t actualSize{0};
    InferenceClient* client = new InferenceClient(grpc::CreateChannel( "localhost:50051", grpc::InsecureChannelCredentials()));
    while (true) {

      responses.clear();
      {
        std::unique_lock<std::mutex> lck(mMtx);
        mCondVar.wait(lck, [&]() { return !mIdQueue.empty() || mDone; });

        if (mDone) {
          break;
        }
        
        actualSize = mIdQueue.size();
        std::vector<mlperf::QuerySample> my_samples;
        my_samples.reserve(actualSize);
        my_samples.swap(mIdQueue);
        mCondVar.notify_one();


        auto results = client->runItems(ds,my_samples);

        responses.reserve(results.size());
        for(auto result: results){
          mlperf::QuerySampleResponse res = {result.id, reinterpret_cast<std::uintptr_t>(result.result), result.size*sizeof(result.result[0])};
          responses.push_back(res);
        }
      }
      mlperf::QuerySamplesComplete(responses.data(), actualSize);

    }
  }


  std::string mName{"QueueSUT"};
  std::vector<std::thread> mThreads;
  std::vector<mlperf::QuerySample> mIdQueue;
  std::mutex mMtx;
  std::condition_variable mCondVar;
  bool mDone{false};
  Dataset &ds;

};

class MultiBasicSUT : public mlperf::SystemUnderTest {
 public:
  MultiBasicSUT(int numThreads, Dataset &ds_)
      : mNumThreads(numThreads), ds(ds_) {

    for (int i = 0; i < mNumThreads; ++i) {
      mThreads.emplace_back(&MultiBasicSUT::startIssueThread, this, i);
    }
  }
  ~MultiBasicSUT() override {
    for (auto& thread : mThreads) {
      thread.join();
    }
  }
  const std::string& Name() const override { return mName; }
  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
    InferenceClient* client = mThreadMap[std::this_thread::get_id()];
    auto results = client->runItems(ds,samples);

    std::vector<mlperf::QuerySampleResponse> responses;
    responses.reserve(results.size());
    for(auto result: results){
      mlperf::QuerySampleResponse res = {result.id, reinterpret_cast<std::uintptr_t>(result.result), result.size*sizeof(result.result[0])};
      responses.push_back(res);
    }
    mlperf::QuerySamplesComplete(responses.data(), responses.size());
  }
  void FlushQueries() override {}
  void ReportLatencyResults(
      const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override{};

 private:
  void startIssueThread(int thread_idx) {
    {
      std::lock_guard<std::mutex> lock(mMtx);
      mThreadMap[std::this_thread::get_id()] = new InferenceClient(grpc::CreateChannel( "localhost:50051", grpc::InsecureChannelCredentials()));;
    }
    mlperf::RegisterIssueQueryThread();
  }

  int mNumThreads{0};
  std::string mName{"MultiBasicSUT"};
  std::mutex mMtx;
  std::vector<std::thread> mThreads;
  std::map<std::thread::id, InferenceClient*> mThreadMap;
  Dataset &ds;
};

int main(int argc, char** argv) {
  assert(argc >= 2 && "Need to pass input dir");

  bool useQueue{false};
  int maxSize{1};
  bool server_coalesce_queries{false};
  int num_issue_threads{0};
  bool grpcCall{true};
  int target_qps{1};
  if (argc >= 3) {
    grpcCall = std::stoi(argv[2]) != 0;
  }
  if (argc >= 4) {
    target_qps = std::stoi(argv[3]);
    std::cout << "target_qps = " << target_qps << std::endl;
  }
  if (argc >= 5) {
    useQueue = std::stoi(argv[4]) != 0;
  }
  if (argc >= 6) {
    num_issue_threads = std::stoi(argv[5]);
  }

  Dataset ds(argv[1],grpcCall);


  QSL qsl(ds);
  std::unique_ptr<mlperf::SystemUnderTest> sut;

  // Configure the test settings
  mlperf::TestSettings testSettings;
  testSettings.scenario = mlperf::TestScenario::Server;
  testSettings.mode = mlperf::TestMode::PerformanceOnly;
  testSettings.server_target_qps = target_qps;
  testSettings.server_target_latency_ns = 10000000;  // 10ms
  testSettings.server_target_latency_percentile = 0.99;
  testSettings.min_duration_ms = 60000;
  testSettings.min_query_count = 10;
  testSettings.server_max_async_queries = -1;
  testSettings.server_coalesce_queries = server_coalesce_queries;
  std::cout << "testSettings.server_coalesce_queries = "
            << (server_coalesce_queries ? "True" : "False") << std::endl;
  // testSettings.server_num_issue_query_threads = num_issue_threads;
  std::cout << "num_issue_threads = " << num_issue_threads << std::endl;

  // Configure the logging settings
  mlperf::LogSettings logSettings;
  logSettings.log_output.outdir = "build";
  logSettings.log_output.prefix = "mlperf_log_";
  logSettings.log_output.suffix = "";
  logSettings.log_output.prefix_with_datetime = false;
  logSettings.log_output.copy_detail_to_stdout = false;
  logSettings.log_output.copy_summary_to_stdout = true;
  logSettings.log_mode = mlperf::LoggingMode::AsyncPoll;
  logSettings.log_mode_async_poll_interval_ms = 1000;
  logSettings.enable_trace = true;

  // Choose SUT
  if (num_issue_threads == 0) {
    if (useQueue) {
      std::cout << "Using QueueSUT with " << num_issue_threads
                << " complete threads" << std::endl;
      sut.reset(new QueueSUT(num_issue_threads, ds));
    } else {
      testSettings.mode = mlperf::TestMode::AccuracyOnly;
      std::cout << "Using BasicSUT" << std::endl;
      sut.reset(new BasicSUT(ds));
    }
  } else {
    if (useQueue) {
      std::cout << "Using QueueSUT with " << num_issue_threads
                << " complete threads" << std::endl;
      // std::cerr << "!!!! MultiQueueSUT is NOT implemented yet !!!!"
      //           << std::endl;
      sut.reset(new QueueSUT(num_issue_threads, ds));
      // sut.reset(new MultiQueueSUT(num_issue_threads, numCompleteThreads,
      // maxSize));
    } else {
      std::cout << "Using MultiBasicSUT" << std::endl;
      testSettings.server_num_issue_query_threads = num_issue_threads;
      sut.reset(new MultiBasicSUT(num_issue_threads, ds));
    }
  }

  // Start test
  std::cout << "Start test..." << std::endl;
  mlperf::StartTest(sut.get(), &qsl, testSettings, logSettings);
  std::cout << "Test done. Clean up SUT..." << std::endl;
  sut.reset();
  std::cout << "Done!" << std::endl;
  InferenceClient* client = new InferenceClient(grpc::CreateChannel( "localhost:50051", grpc::InsecureChannelCredentials()));
  client->end();
  return 0;
}
