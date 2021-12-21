#include "mlperf_sut.h"
// #include "run_samples.h"
// #include "OnnxMlirRuntime.h"

// extern "C"{
// OMTensorList *run_main_graph(OMTensorList *);
// }

// using grpc::Channel;
// using grpc::ClientContext;
// using grpc::Status;
// using helloworld::InferenceService;
// using helloworld::InferenceResponse;
// using helloworld::InferenceRequest;



namespace grpc_tests {

// std::vector<InferenceResult> runItems( Dataset &ds, const std::vector<mlperf::QuerySample>& samples ){

//   std::string target_str = "localhost:50051";
//   InferenceServiceClient InferenceService(
//       grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));

//   std::vector<InferenceResult> results;

//   int count = 0;
//   for(auto s: samples){
//     InferenceService.InferenceSample(ds, s, results);
    
//     // std::vector<float> f = ds.getImageData(s.index);
//     // int64_t* shape = ds.shape;
//     // int64_t rank = ds.rank;
//     // float *x1Data = f.data();
//     // OMTensor *x1 = omTensorCreate(x1Data, shape, rank, ONNX_TYPE_FLOAT);

//     // OMTensor *list[1] = {x1};
//     // OMTensorList *input = omTensorListCreate(list,1); //new OMTensorList(list,1);
//     // OMTensorList *outputList = run_main_graph(input);
//     // OMTensor *y = omTensorListGetOmtByIndex(outputList,0); // outputList->_omts[0];
//     // //float *prediction = (float *)y->_alignedPtr;
//     // float *prediction = (float *)omTensorGetDataPtr(y);
//     // // int resultsize = y->_shape[1];
//     // int64_t *output_shape = omTensorGetShape(y);

//     // int resultsize = 1;
//     // for(int i = 0; i < omTensorGetRank(y); i++){
//     //   resultsize *= output_shape[i];
//     // }

//     // results.push_back({s.id,prediction,resultsize});

//   }

//   return results;
// }

std::vector<InferenceResult> runItems( Dataset &ds, const std::vector<mlperf::QuerySample>& samples ){
  std::vector<InferenceResult> results;
  int count = 0;
  for(auto s: samples){
    std::vector<float> input = ds.getImageData(s.index);
    // std::vector<float> output = InferenceService.InferenceSample(input, ds.shape, ds.rank);
    std::vector<float> output = ds.client.runOne(input, ds.shape, ds.rank);
    results.push_back({s.id,output.data(),output.size()});
  }
  return results;
}

  SystemUnderTestSingleStream::SystemUnderTestSingleStream(Dataset &ds):ds(ds){}
  const std::string&  SystemUnderTestSingleStream::Name() const { return name_; }
  void SystemUnderTestSingleStream::IssueQuery(const std::vector<mlperf::QuerySample>& samples) {
    auto results = runItems(ds,samples);
    std::vector<mlperf::QuerySampleResponse> responses;
    responses.reserve(results.size());
    for(auto result: results){
      mlperf::QuerySampleResponse res = {result.id, reinterpret_cast<std::uintptr_t>(result.result), result.size*sizeof(result.result[0])};
      responses.push_back(res);
    }
    mlperf::QuerySamplesComplete(responses.data(), responses.size());
  }




  SystemUnderTestStdAsync::SystemUnderTestStdAsync(Dataset &ds):ds(ds) { futures_.reserve(1000000); }
  const std::string& SystemUnderTestStdAsync::Name() const { return name_; }
  void SystemUnderTestStdAsync::IssueQuery(const std::vector<mlperf::QuerySample>& samples) {
    futures_.emplace_back(std::async(std::launch::async, [samples,&ds=ds] {
      auto results = runItems(ds,samples);
      std::vector<mlperf::QuerySampleResponse> responses;
      responses.reserve(results.size());
      for(auto result: results){
        mlperf::QuerySampleResponse res = {result.id, reinterpret_cast<std::uintptr_t>(result.result), result.size*sizeof(result.result[0])};
        responses.push_back(res);
      }
      mlperf::QuerySamplesComplete(responses.data(), responses.size());
    }));
  }



  SystemUnderTestPool::SystemUnderTestPool(Dataset &ds, size_t thread_count):ds(ds) {
    samples_.reserve(kReserveSampleSize);
    thread_count_ = thread_count;
    next_poll_time_ = std::chrono::high_resolution_clock::now() + poll_period_;
    for (size_t i = 0; i < thread_count_; i++) {
      threads_.emplace_back(&SystemUnderTestPool::WorkerThread, this);
    }
  }

  SystemUnderTestPool::~SystemUnderTestPool() {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      keep_workers_alive_ = false;
    }
    cv_.notify_all();
    for (auto& thread : threads_) {
      thread.join();
    }
  }

  const std::string& SystemUnderTestPool::Name() const { return name_; }

  void SystemUnderTestPool::IssueQuery(const std::vector<mlperf::QuerySample>& samples) {
    std::unique_lock<std::mutex> lock(mutex_);
    samples_.insert(samples_.end(), samples.begin(), samples.end());
  }

  void SystemUnderTestPool::WorkerThread() {
    std::vector<mlperf::QuerySample> my_samples;
    my_samples.reserve(kReserveSampleSize);
    std::unique_lock<std::mutex> lock(mutex_);
    while (keep_workers_alive_) {
      next_poll_time_ += poll_period_;
      auto my_wakeup_time = next_poll_time_;
      cv_.wait_until(lock, my_wakeup_time,
                     [&]() { return !keep_workers_alive_; });
      my_samples.swap(samples_);
      lock.unlock();

      auto results = runItems(ds,my_samples);
      std::vector<mlperf::QuerySampleResponse> responses;
      responses.reserve(results.size());
      for(auto result: results){
        mlperf::QuerySampleResponse res = {result.id, reinterpret_cast<std::uintptr_t>(result.result), result.size*sizeof(result.result[0])};
        responses.push_back(res);
      }
      mlperf::QuerySamplesComplete(responses.data(), responses.size());

      lock.lock();
      my_samples.clear();
    }
  }

  void QueueSUT::CompleteThread(int threadIdx) {
    std::vector<mlperf::QuerySample> my_samples;
    my_samples.reserve(1024 * 1024);
    while (true) {
      {
        std::unique_lock<std::mutex> lck(mMtx);
        mCondVar.wait(lck, [&]() { return !samples_.empty() || mDone; });

        if (mDone) {
          break;
        }
        
        my_samples.swap(samples_);
        mCondVar.notify_one();
        auto results = runItems(ds,my_samples);
        std::vector<mlperf::QuerySampleResponse> responses;
        responses.reserve(results.size());
        for(auto result: results){
          mlperf::QuerySampleResponse res = {result.id, reinterpret_cast<std::uintptr_t>(result.result), result.size*sizeof(result.result[0])};
          responses.push_back(res);
        }
        my_samples.clear();
        mlperf::QuerySamplesComplete(responses.data(), responses.size());
      }
      
    }
  }


  // Dataset::Dataset(std::string imagePath_, InferenceServiceClient &client):client(client) {
  //   imagePath = imagePath_;
  //   readImageList(imagePath);
  //   client.start();
  // }

  void Dataset::readImage(std::string imageName, std::vector<float> &imageData)
  {
      std::string currentImagePath = imagePath + "/" + imageName;
      std::ifstream fp(currentImagePath, std::ios::in | std::ios::binary);

      if (!fp.is_open()) {
        printf("read image %s error\n", imageName);
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


  int Dataset::readImageList(std::string imagePath) {

    std::ifstream fp(imagePath +"/val_map.txt");

    if (!fp.is_open()) {
      printf("read val_map error\n");
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

  void Dataset::loadImageData(long index){
    std::vector<float> imageData;
    readImage(imageList[index], imageData);
    imageInMemory[index] = imageData;
  }

  int Dataset::getImageLabel(int index){
    return imageLabels[index];
  }

  long Dataset::getImageCount(){
    return imageList.size();
  }

  std::vector<float> Dataset::getImageData(long index){
    return imageInMemory[index];
  }

}