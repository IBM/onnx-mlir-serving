#pragma once
#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <atomic>
#include <future>

#include <chrono>
//#include <condition_variable>
//#include <thread>
//#include <functional>
#include <stdexcept>
// #include "date.h"
//
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include <grpc/support/log.h>
#include <grpcpp/grpcpp.h>

#ifdef BAZEL_BUILD
#include "examples/protos/inference.grpc.pb.h"
#else
#include "inference.grpc.pb.h"
#endif

using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerCompletionQueue;
using grpc::ServerContext;
using grpc::Status;
using inference::InferenceService;
using inference::InferenceResponse;
using inference::InferenceRequest;
using inference::EndRequest;
using inference::EndResponse;

#include "OnnxMlirRuntime.h"
extern "C"{
OMTensorList *run_main_graph(OMTensorList *);
}

namespace std
{

class CallData
{
  
	public:
		enum ServiceType {
			inference = 0,
			end = 1
		};
  public:
    CallData(InferenceService::AsyncService* service, ServerCompletionQueue* cq, ServiceType s_type)
        : service_(service), cq_(cq), s_type_(s_type), responder_(&ctx_), endResponder_(&ctx_), status_(CREATE){
      Proceed(NULL);
    }

    void Proceed(void *threadpool);

    float *getRequestData(){
      vector<float> results(request_.data().begin(),request_.data().end());
      return results.data();
    }

    void sendBack(float* data, int size){
      reply_.mutable_data()->Add(data, data+size);
      status_ = FINISH;
      responder_.Finish(reply_, Status::OK, this);
    }

    chrono::high_resolution_clock::time_point now;

  private:
  InferenceService::AsyncService* service_;
  ServerCompletionQueue* cq_;
  ServerContext ctx_;
  InferenceRequest request_;
  EndRequest endRequest_;
  InferenceResponse reply_;
  EndResponse endReply_;
  ServerAsyncResponseWriter<InferenceResponse> responder_;
  ServerAsyncResponseWriter<EndResponse> endResponder_;
  enum CallStatus { CREATE, PROCESS, FINISH };
  CallStatus status_;  // The current serving state.
  ServiceType s_type_;
};


#define  THREADPOOL_MAX_NUM 16
#define  MODEL_MAX_NUM 128

class AIUThreadpool
{
  using Task = function<void()>;
  vector<thread> _pool; 
  vector<thread> timecheck;
  vector<size_t> logs; 
  queue<CallData*> _inference_data;
  mutex _lock;
  mutex log_lock;
  condition_variable _task_cv;
  atomic<bool> _run{ true };
  atomic<int>  _idlThrNum{ 0 };
  int wait = 0;

  public:
    AIUThreadpool(unsigned short size, int wait_){ 
      wait = wait_;
      addThread(size);
      timeThread();
    }
    ~AIUThreadpool()
    {
        _run=false;
        _task_cv.notify_all();
        for (thread& thread : _pool) {
            //thread.detach();
            if(thread.joinable())
                thread.join(); 
        }
    }

public:

    void addCallData(CallData *data){
      int size = _inference_data.size();
      {
        lock_guard<std::mutex> lock(_lock);
        _inference_data.emplace(data);
        size = _inference_data.size();
        // cout <<  "inferende size add " << _inference_data.size() << endl;
      }

      if(size >= MODEL_MAX_NUM){
        _task_cv.notify_one();
      }

    }

    void printlogs(){
      cout << "end" << endl;
      for(int i; i < logs.size(); i++){
        cout << logs[i] << endl;
      }
    }

    void forceRun(){
      _task_cv.notify_one();
    }


    int idlCount() { return _idlThrNum; }

    int thrCount() { return _pool.size(); }

    void timeThread(){
      timecheck.emplace_back([this]{
        while(_run){
          bool check = false;
          int size = 0;
          double d = 0;
          {
            lock_guard<std::mutex> lock(_lock);
            // cout <<  "check " << endl;
            size = _inference_data.size();
            if(size>0){
              chrono::high_resolution_clock::time_point pnow = _inference_data.front()->now;
              chrono::high_resolution_clock::time_point now = chrono::high_resolution_clock::now();
              // cout <<  "p durarion " << now -pnow << endl;
              d = std::chrono::duration<double, std::nano>(now -pnow).count();
              check = d > wait && idlCount()>0;
              // if (std::chrono::duration<double, std::nano>(now -pnow).count() > wait && idlCount()>0){
              //   cout <<  "force notify " <<_inference_data.size()<< endl;
              //   forceRun();
              // }
            }
          }

          if(check){
            // cout <<  "force notify " <<size << " d " << d<< endl;
            forceRun();
          }else{
            // cout <<  "not force notify " <<size << " d " << d<< endl;
          }
          this_thread::sleep_for(std::chrono::milliseconds(1));
        }
      });

    }
    


    void addThread(unsigned short size)
    {
        for (; _pool.size() < THREADPOOL_MAX_NUM && size > 0; --size)
        {  
            _pool.emplace_back( [this]{ 
                while (_run)
                {
                    
                    vector<CallData*> my_queue;
                    my_queue.reserve(MODEL_MAX_NUM);
                    {
                        unique_lock<mutex> lock{ _lock };
                        _task_cv.wait(lock, [this]{
                                return !_run || !_inference_data.empty();
                        }); // wait until _inference_data is not empty
                        // cout <<  "inferende size " << _inference_data.size() << endl;
                        if (!_run && _inference_data.empty())
                            return;
                        
                        my_queue.clear();
                        
                        int count = 0;
                        int totalsize = _inference_data.size();
                        while(count<MODEL_MAX_NUM && count<totalsize){
                            my_queue.push_back(_inference_data.front());
                            _inference_data.pop();
                            count ++;
                        }
                    }
                    _idlThrNum--;
                    runInference(my_queue); //run AI model
                    // cout <<  "inferende done " << endl;
                    _idlThrNum++;
                }
            });
            _idlThrNum++;
        }
    }

    void runInference(vector<CallData*> calldata_list){

      size_t batchsize = calldata_list.size();

      int64_t rank = 3;
      int64_t shape[3] = {7,batchsize,204};
      int64_t totalsize = 7* batchsize * 204;
      float *x1Data = (float*)calloc(totalsize, sizeof(float));
      // float x1Data[totalsize] = {0};

      for(size_t i = 0; i < batchsize; i ++){
        // if(i<calldata_list.size()){
          float *curr = calldata_list[i]->getRequestData();

          for(size_t j = 0; j < 7; j++){
            for(size_t k = 0; k< 204; k++){
              x1Data[(j*batchsize+i)*204+k] = curr[(j*204)+k];
            }
          }
        // }
      }

      OMTensor *x1 = omTensorCreate(x1Data, shape, rank, ONNX_TYPE_FLOAT);
      OMTensor *list[1] = {x1};
      OMTensorList *input = omTensorListCreate(list,1);
      OMTensorList *outputList = run_main_graph(input);
      

      OMTensor *y = omTensorListGetOmtByIndex(outputList,0);
      float *prediction = (float *)omTensorGetDataPtr(y);
      int64_t *output_shape = omTensorGetShape(y);
      int resultsize = 1;
      for(int i = 0; i < omTensorGetRank(y); i++){
        resultsize *= output_shape[i];
      }

      float* singleResult = (float*)calloc(batchsize*7, sizeof(float));
      // float singleResult[MODEL_MAX_NUM*7] = {0};
      float* start = singleResult;
      for(size_t i = 0;i < batchsize; i++ ){
        // if(i<calldata_list.size()){
          for(size_t j = 0; j < 7; j++){
            start[j] = prediction[j*batchsize+i];
          }
          calldata_list[i]->sendBack(start,7);
        // }
        start += 7;
      }

      free(singleResult);
      free(x1Data);
      omTensorDestroy(x1);
      omTensorDestroy(y);

      lock_guard<std::mutex> lock(log_lock);
      logs.push_back(batchsize);

    }

};

}

#endif  