#pragma once
#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <atomic>
#include <future>

#include <fstream>
#include <chrono>
//#include <condition_variable>
//#include <thread>
//#include <functional>
#include <stdexcept>
#include <sys/prctl.h>
// #include "date.h"
//
#include <iostream>
#include <sstream>
#include <memory>
#include <string>
// #include <thread>
#include <pthread.h>

#include <grpc/support/log.h>
#include <grpcpp/grpcpp.h>


#include "inference.grpc.pb.h"
// #include "onnx.pb.h"


#include "aiu_thread_pool.h"

using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerCompletionQueue;
using grpc::ServerContext;
using grpc::Status;
using inference::InferenceService;
using inference::InferenceResponse;
using inference::InferenceRequest;
using inference::PrintStatisticsRequest;
using inference::PrintStatisticsResponse;

// #include "OnnxMlirRuntime.h"
// extern "C"{
// OMTensorList *run_main_graph(OMTensorList *);
// }





using std::vector;
using std::queue;
using std::mutex;
using std::condition_variable;
using std::atomic;
using std::string;
using std::thread;
using std::lock_guard;
using std::unique_lock;
using std::chrono::high_resolution_clock;


#define  THREADPOOL_MAX_NUM 20


class CallData:public AbstractCallData
{
  
	public:
		enum ServiceType {
			inference = 0,
			printStatistics = 1
		};
  public:
    CallData(InferenceService::AsyncService* service, ServerCompletionQueue* cq, ServiceType s_type)
        : service_(service), cq_(cq), responder_(&ctx_), printStatisticsResponder_(&ctx_), s_type_(s_type), status_(CREATE){
      Proceed(NULL);
    }

    void Proceed(void *threadpool);

    InferenceRequest getRequestData(){
      return request_;
    }

    void sendBack(float* data, int size){
      reply_.mutable_data()->Add(data, data+size);
      // reply_.set_id(request_.id());
      status_ = FINISH;
      responder_.Finish(reply_, Status::OK, this);
    }

    // high_resolution_clock::time_point now;

  private:
    InferenceService::AsyncService* service_;
    ServerCompletionQueue* cq_;
    ServerContext ctx_;
    InferenceRequest request_;
    PrintStatisticsRequest printStatisticsRequest_;
    InferenceResponse reply_;
    PrintStatisticsResponse printStatisticsReply_;
    ServerAsyncResponseWriter<InferenceResponse> responder_;
    ServerAsyncResponseWriter<PrintStatisticsResponse> printStatisticsResponder_;
    ServiceType s_type_;
    enum CallStatus { CREATE, PROCESS, FINISH };
    CallStatus status_;  // The current serving state.

};

class DLCModelManager{
  public:
    DLCModelManager(int batch_size, int thread_num, int wait_time):tpool_(thread_num),checkBatchingThread_([this]{checkBatching();}){
      batch_size_ = batch_size;
      wait_time_  = wait_time;
    }
    ~DLCModelManager(){ 
      run_=0;
      if(checkBatchingThread_.joinable()){
        checkBatchingThread_.join();
      }
    }
    int AddModel(CallData *data){
      const char* model_name = data->getRequestData().model_name().c_str();
      DLCModel* model = NULL;
      {
        lock_guard<mutex> lock(lock_);
        model = Get_model(model_name);
      }

      if(model == NULL){
        data->sendBack(NULL, 0);
        return 0;
      }

      //no batching, add task to thread pool right now
      if(model->batching == 0)
      {
        // std::cout<<"no batching"<<std::endl;
        tpool_.AddTask(model->Perpare_and_run(data));
      }
      //else add data to inference queue, wait batching
      else
      { 
        // std::cout<<"batching"<<std::endl;
        model->AddInferenceData(data);
      }

      return 1;

    }
    void PrintLogs(){
      tpool_.PrintLogs();
    }

  private:
    void checkBatching(){
      while(run_){
        {
          lock_guard<mutex> lock(lock_);
          for (size_t i = 0; i < models_.size(); i++){
            DLCModel* model = models_.at(i);
            if(model->batching && model->Ready(wait_time_)){
              tpool_.AddTask(model->Perpare_and_run(batch_size_));
            }
          }
        }
        std::this_thread::sleep_for(std::chrono::nanoseconds((int)(10000)));
      }
    }

    DLCModel* Get_model(const char* model_name){
      DLCModel* model = NULL;
      //get model from exist model queue
      for (size_t i = 0; i < models_.size(); i++)
      {
        if(strcmp(model_name, models_[i]->model_name)==0){
          model = models_[i];
          return model;
        }
      }

      //create new model when not find
      std::cout<<"create new model " << model_name <<std::endl;
      model = new DLCModel(model_name);
      if(model->model_name[0] == 0){
        return NULL;
      }
      models_.emplace_back(model);
      
      return model;

    }

    std::vector<DLCModel*> models_;
    AIUThreadPool tpool_;
    std::thread checkBatchingThread_;
    std::mutex lock_;
    int run_ = 1;
    int batch_size_;
    int wait_time_;
};


class ServerImpl final {
 public:
  ~ServerImpl() {
    server_->Shutdown();
    // Always shutdown the completion queue after the server.
    cq_->Shutdown();
  }

  ServerImpl(int batch_size, int threadNum_, int wait):modelManager_(batch_size, threadNum_, wait){
  }

  void Run();

 private:
  void HandleRpcs(int i);

  std::shared_ptr<ServerCompletionQueue> cq_;
  InferenceService::AsyncService service_;
  // AIUThreadPool tpool;
  DLCModelManager modelManager_;
  std::shared_ptr<Server> server_;
  vector<std::thread> async_threads;
  mutex  mtx_;
};


#endif  
