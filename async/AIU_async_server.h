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
#include <sstream>
#include <memory>
#include <string>
#include <thread>

#include <grpc/support/log.h>
#include <grpcpp/grpcpp.h>


#include "inference.grpc.pb.h"

#include "DLCModelLoader.h"

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

// #include "OnnxMlirRuntime.h"
// extern "C"{
// OMTensorList *run_main_graph(OMTensorList *);
// }


std::chrono::high_resolution_clock::time_point  originTime(std::chrono::seconds(1646319840));


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
#define  MODEL_MAX_NUM 10

DLCModelLoader modelLoder;

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

    InferenceRequest getRequestData(){
      return request_;
    }

    void sendBack(float* data, int size){
      reply_.mutable_data()->Add(data, data+size);
      reply_.set_id(request_.id());
      status_ = FINISH;
      responder_.Finish(reply_, Status::OK, this);
    }

    high_resolution_clock::time_point now;

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


class AIUThreadpool
{
  using Task = std::function<void()>;
  vector<thread> _pool; 
  vector<thread> timecheck;
  queue<CallData*> _inference_data;
  mutex _lock;
  mutex _log_mutex;
  condition_variable _task_cv;
  atomic<bool> _run{ true };
  atomic<int>  _idlThrNum{ 0 };
  int wait = 0;
  std::stringstream _log_stream;

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
        lock_guard<mutex> lock(_lock);
        _inference_data.push(data);
        size = _inference_data.size();
        if((size >= MODEL_MAX_NUM || wait == 0 )  ){
          _task_cv.notify_one();
        }
      }
    }

    void printlogs(){
      std::cout << _log_stream.str() << std::endl;
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
            lock_guard<mutex> lock(_lock);

            size = _inference_data.size();
            if(size>0 && wait !=0){
              high_resolution_clock::time_point pnow = _inference_data.front()->now;
              high_resolution_clock::time_point now = high_resolution_clock::now();
              d = std::chrono::duration<double, std::nano>(now -pnow).count();
              double w = wait * (float)size;
              if(d >= w && idlCount()>0){
                forceRun();
              }
            } 
          }

          std::this_thread::sleep_for(std::chrono::nanoseconds(100));
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
                    high_resolution_clock::time_point pnow;
                    high_resolution_clock::time_point now;
                    vector<CallData*> my_queue;
                    my_queue.reserve(MODEL_MAX_NUM);
                    {
                      unique_lock<mutex> lock{ _lock };
                      double d = 0;

                      do{
                        _task_cv.wait(lock, [this]{ return !_run || !_inference_data.empty(); }); // wait until _inference_data is not empty
                        pnow = _inference_data.front()->now;
                        now = high_resolution_clock::now();
                        d = std::chrono::duration<double, std::nano>(now -pnow).count();
                      }while(_inference_data.size()<MODEL_MAX_NUM && d <wait);


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

                      lock.unlock();
                      lock_guard<mutex> log_lock(_log_mutex);
                      _log_stream << std::this_thread::get_id() << ",wake up," << _inference_data.size() << "," << d << ",";
                      _log_stream << std::to_string(std::chrono::duration<double, std::nano>(pnow -originTime).count()) << ",";
                      _log_stream << std::to_string(std::chrono::duration<double, std::nano>(now -originTime).count()) << std::endl;
                    }
                    _idlThrNum--;
                    runInference(my_queue); //run AI model
                    _idlThrNum++;
                }
            });
            _idlThrNum++;
        }
    }

    void runInference(vector<CallData*> calldata_list){

      high_resolution_clock::time_point pnow = high_resolution_clock::now();

      size_t batchsize = calldata_list.size();

      int64_t rank = 3;
      int64_t shape[3] = {7,batchsize,204};
      int64_t totalsize = 7* batchsize * 204;
      float *x1Data = (float*)calloc(totalsize, sizeof(float));

      for(size_t i = 0; i < batchsize; i ++){
        auto curr = calldata_list[i]->getRequestData();
        for(size_t j = 0; j < 7; j++){
          for(size_t k = 0; k< 204; k++){
            x1Data[(j*batchsize+i)*204+k] = curr.data((j*204)+k);
          }
        }
      }


      // high_resolution_clock::time_point pnow = high_resolution_clock::now();

      // OMTensor *x1 = omTensorCreate(x1Data, shape, rank, ONNX_TYPE_FLOAT);
      // OMTensor *list[1] = {x1};
      // OMTensorList *input = omTensorListCreate(list,1);
      // OMTensorList *outputList = run_main_graph(input);

      OMTensor *y = modelLoder.RunModel(x1Data, shape, rank, ONNX_TYPE_FLOAT);; //omTensorListGetOmtByIndex(outputList,0);
      float *prediction = (float *)omTensorGetDataPtr(y);
      int64_t *output_shape = omTensorGetShape(y);
      int resultsize = 1;
      for(int i = 0; i < omTensorGetRank(y); i++){
        resultsize *= output_shape[i];
      }

      float* singleResult = (float*)calloc(batchsize*7, sizeof(float));

      // chrono::high_resolution_clock::time_point now = chrono::high_resolution_clock::now();

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

      high_resolution_clock::time_point now = high_resolution_clock::now();
      double d = std::chrono::duration<double, std::nano>(now -pnow).count();

      free(singleResult);
      free(x1Data);
      // free(prediction);
      // omTensorDestroy(x1);
      omTensorDestroy(y);



      lock_guard<mutex> lock(_log_mutex);

      //thread id, action, size, dur, start time, now time
      _log_stream << std::this_thread::get_id() << ",inference," << batchsize << "," << d << ",";
      _log_stream << std::to_string(std::chrono::duration<double, std::nano>(pnow -originTime).count()) << ",";
      _log_stream << std::to_string(std::chrono::duration<double, std::nano>(now -originTime).count()) << std::endl;


    }

};



#endif  