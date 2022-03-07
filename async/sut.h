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

// #include "OnnxMlirRuntime.h"

// extern "C"{
// OMTensorList *run_main_graph(OMTensorList *);
// }

#include <grpc/support/log.h>
#include <grpcpp/grpcpp.h>


#include "inference.grpc.pb.h"
#include "DLCModelLoader.h"

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


DLCModelLoader modelLoder;

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
    void* got_tag;
    bool ok = false;
    GPR_ASSERT(cq.Next(&got_tag, &ok));
    GPR_ASSERT(got_tag == (void*)1);
    std::cout << "end done"<<std::endl;
  }

  void AsyncCompleteRpc() {
    void* got_tag;
    bool ok = false;

    while (cq_.Next(&got_tag, &ok)) {
      // std::cout<<std::this_thread::get_id()<<" get call re"<<std::endl;
      AsyncClientCall* call = static_cast<AsyncClientCall*>(got_tag);
      GPR_ASSERT(ok);

      // if (call->status.ok()){
      std::vector<float> out(call->reply.data().begin(), call->reply.data().end());
      results_.push_back({call->reply.id(),out.data(),out.size()});

      mlperf::QuerySampleResponse res = {call->reply.id(), reinterpret_cast<std::uintptr_t>(out.data()), out.size()*sizeof(out.data()[0])};
      mlperf::QuerySamplesComplete(&res, 1);
      // }
      // else
      //   std::cout << "RPC failed" << std::endl;

      if (!call->status.ok()){
         std::cout <<std::this_thread::get_id() << " RPC failed " << call->reply.id() << std::endl;
      }

      // Once we're complete, deallocate the call object.
      delete call;
      if (results_.size() >= current_size){
        break;
      }
    }
  }

  std::vector<InferenceResult> runItems( Dataset &ds, const std::vector<mlperf::QuerySample>& samples ){
    // std::cout <<  "inference call " << std::endl;
    if(ds.grpcCall){
      std::vector<InferenceResult> results;
      current_size = samples.size();
      results_.clear();
      std::thread t (&InferenceClient::AsyncCompleteRpc, this);
      for(auto sample: samples){
        // std::cout <<std::this_thread::get_id() << " id " << sample.id <<std::endl;
        auto input = ds.getImageData(sample.index);
        Inference(input, ds.shape, ds.rank, sample.id);
        // std::vector<float> output = Inference(input, ds.shape, ds.rank, sample.id);
        // results.push_back({sample.id,output.data(),output.size()});
      }
      // std::cout <<std::this_thread::get_id() << " query " << current_size <<std::endl;
      t.join();
      // std::cout <<std::this_thread::get_id() << " done " << current_size <<std::endl;
      return results_;
    }else{
      return InferenceLocal(ds,samples);
    }
  }

  void Inference(std::vector<float> &input_data, int64_t* shape, int64_t rank, unsigned long id) {
    InferenceRequest request;
    request.mutable_data()->Add(input_data.begin(), input_data.end());
    request.mutable_shape()->Add(shape, shape+rank);
    request.set_id(id);
    // InferenceResponse response;
    // std::cout<<"get call re1"<<std::endl;

    AsyncClientCall* call = new AsyncClientCall;
    call->response_reader =
        stub_->PrepareAsyncInference(&call->context, request, &cq_);

    call->response_reader->StartCall();
    call->response_reader->Finish(&call->reply, &call->status, (void*)call);

  }

  std::vector<InferenceResult> InferenceLocal( Dataset &ds, const std::vector<mlperf::QuerySample>& samples ){

    std::vector<InferenceResult> results;
    for(auto s: samples){

    std::vector<float> f = ds.getImageData(s.index);

    int64_t* shape = ds.shape;
    int64_t rank = ds.rank;
    float *x1Data = f.data();

    OMTensor *y = modelLoder.RunModel(x1Data, shape, rank, ONNX_TYPE_FLOAT);
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
   struct AsyncClientCall {
    InferenceResponse reply;
    ClientContext context;
    Status status;
    std::unique_ptr<ClientAsyncResponseReader<InferenceResponse>> response_reader;
  };
  std::unique_ptr<InferenceService::Stub> stub_;
  std::vector<InferenceResult> results_;
  CompletionQueue cq_;
  int current_size = 0;
};


class QSL : public mlperf::QuerySampleLibrary {
 public:
  QSL(Dataset &ds):ds(ds){}
  ~QSL() = default;
  const std::string& Name() const override { return name_; }
  size_t TotalSampleCount() override { return ds.getImageCount(); }
  size_t PerformanceSampleCount() override { return 60; }

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

        InferenceClient* client = new InferenceClient(grpc::CreateChannel( "localhost:50051", grpc::InsecureChannelCredentials()));
        auto results = client->runItems(ds,my_samples);

        // responses.reserve(results.size());
        // // std::cout << results.size() <<std::endl;
        // for(auto result: results){
        //   mlperf::QuerySampleResponse res = {result.id, reinterpret_cast<std::uintptr_t>(result.result), result.size*sizeof(result.result[0])};
        //   responses.push_back(res);
        // }
        delete client;
      }
      // mlperf::QuerySamplesComplete(responses.data(), actualSize);

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