#ifndef MLPERF_H
#define MLPERF_H 
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wtype-limits"
#pragma GCC diagnostic ignored "-Wformat="
#include <future>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <map>

#include "loadgen.h"
#include "query_sample_library.h"
#include "system_under_test.h"
#include "test_settings.h"

#include "run_samples.h"
// #include <memory>

// #include <grpcpp/grpcpp.h>

// #ifdef BAZEL_BUILD
// #include "examples/protos/helloworld.grpc.pb.h"
// #else
// #include "helloworld.grpc.pb.h"
// #endif

// using grpc::Channel;
// using grpc::ClientContext;
// using grpc::Status;
// using helloworld::InferenceService;
// using helloworld::InferenceResponse;
// using helloworld::InferenceRequest;

namespace grpc_tests {

class Dataset {
 public:
  Dataset(std::string imagePath_, InferenceService2Client &client):client(client){
    imagePath = imagePath_;
    readImageList(imagePath);
  };
  void readImage(std::string imageName, std::vector<float> &imageData);
  int readImageList(std::string imagePath);
  void loadImageData(long index);
  int getImageLabel(int index);
  long getImageCount();
  std::vector<float> getImageData(long index);
  std::int64_t rank;
  std::int64_t* shape;
  InferenceService2Client &client;

  private:
    std::string imagePath;
    std::vector<std::string> imageList;
    std::vector<int> imageLabels;
    std::map<long, std::vector<float>> imageInMemory;
    
};

struct InferenceResult {
  mlperf::ResponseId id;
  float* result;
  int size;
};

// class InferenceServiceClient {
//  public:
//   InferenceServiceClient(std::shared_ptr<Channel> channel)
//       : stub_(InferenceService::NewStub(channel)) {}

//   void InferenceSample(Dataset &ds, mlperf::QuerySample sample, std::vector<InferenceResult> &results){
//     std::vector<float> f = ds.getImageData(sample.index);
//     int64_t* shape = ds.shape;
//     int64_t rank = ds.rank;
//     InferenceRequest request;
//     request.mutable_data()->Add(f.begin(), f.end());
//     request.mutable_shape()->Add(shape, shape+rank);
//     InferenceResponse reply;
//     ClientContext context;
//     Status status = stub_->Inference(&context, request, &reply);
//     float a = 0;
//     if (status.ok()) {
//       // results.add(reply.data.begin(), reply.data.end());
//       std::vector<float> re(reply.data().begin(), reply.data().end());
//       results.push_back({sample.id,re.data(),re.size()});
//       results.push_back({sample.id,reply.mutable_data()->mutable_data(),reply.data_size()});

//     }else{
//       results.push_back({sample.id,&a,1});
//     }
//   }

//  private:
//   std::unique_ptr<InferenceService::Stub> stub_;
// };

class QuerySampleLibraryNull : public mlperf::QuerySampleLibrary {
 public:
  QuerySampleLibraryNull(Dataset &ds):ds(ds){}
  ~QuerySampleLibraryNull() = default;
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

// SUT for single stream
class SystemUnderTestSingleStream : public mlperf::SystemUnderTest {
 public:
  SystemUnderTestSingleStream(Dataset &ds);
  ~SystemUnderTestSingleStream() override = default;
  const std::string& Name() const override;
  void IssueQuery(const std::vector<mlperf::QuerySample>& samples)override;
  void FlushQueries() override {}
  void ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override {}

 private:
  std::string name_{"SingleStream"};
  Dataset &ds;
};

// SUT for server
class SystemUnderTestStdAsync : public mlperf::SystemUnderTest {
 public:
  SystemUnderTestStdAsync(Dataset &ds);
  ~SystemUnderTestStdAsync()override = default;;
  const std::string& Name()const override;
  void IssueQuery(const std::vector<mlperf::QuerySample>& samples)override;
  void FlushQueries() override {}
  void ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override {}

 private:
  std::string name_{"StdAsync"};
  std::vector<std::future<void>> futures_;
  Dataset &ds;
};

// SUT for server or Multi stream
// there is 4 thread running,
//    for Multi stream if multi_stream_max_async_queries > 1
class SystemUnderTestPool : public mlperf::SystemUnderTest {
 public:
  SystemUnderTestPool(Dataset &ds, size_t thread_count);
  ~SystemUnderTestPool();
  const std::string& Name()const override;
  void IssueQuery(const std::vector<mlperf::QuerySample>& samples)override;
  void FlushQueries()override{}
  void ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency>& latencies_ns)override{}

 private:
  void WorkerThread();

  static constexpr size_t kReserveSampleSize = 1024 * 1024;
  const std::string name_{"Pool"};
  size_t thread_count_ = 4;
  const std::chrono::milliseconds poll_period_{1};
  std::chrono::high_resolution_clock::time_point next_poll_time_;

  std::mutex mutex_;
  std::condition_variable cv_;
  bool keep_workers_alive_ = true;
  std::vector<std::thread> threads_;

  std::vector<mlperf::QuerySample> samples_;
  Dataset &ds;
};

class QueueSUT : public mlperf::SystemUnderTest{
  public:
  QueueSUT(Dataset &ds, int numCompleteThreads):ds(ds){
    // Each thread handle at most maxSize at a time.
    // std::cout << "QueueSUT: maxSize = " << maxSize << std::endl;
    // initResponse(numCompleteThreads, maxSize);
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
    samples_.insert(samples_.end(), samples.begin(), samples.end());
    mCondVar.notify_one();
  }
  void FlushQueries() override {}
  void ReportLatencyResults(
      const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override{};

  private:
    void CompleteThread(int threadIdx);
    std::string mName{"QueueSUT"};
    std::vector<std::thread> mThreads;
    std::vector<mlperf::QuerySample> samples_;
    std::mutex mMtx;
    std::condition_variable mCondVar;
    bool mDone{false};
    Dataset &ds;
  };


}
#endif