#pragma once
#ifndef GRPC_SERVER_H
#define GRPC_SERVER_H

#include <vector>
#include <queue>
// #include <atomic>
// #include <future>

#include <fstream>
#include <chrono>
#include <condition_variable>
#include <thread>
#include <pthread.h>
// #include <functional>
#include <stdexcept>
#include <sys/prctl.h>

// #include <iostream>
// #include <sstream>
#include <memory>
#include <string>


#include <grpc/support/log.h>
#include <grpcpp/grpcpp.h>

#include "inference.grpc.pb.h"
// #include "onnx.pb.h"


#include "model_manager.h"

using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerCompletionQueue;
using grpc::ServerContext;
using grpc::Status;
using inference::InferenceRequest;
using inference::InferenceResponse;
using inference::InferenceService;
using inference::PrintStatisticsRequest;
using inference::PrintStatisticsResponse;

// #include "OnnxMlirRuntime.h"
// extern "C"{
// OMTensorList *run_main_graph(OMTensorList *);
// }

using std::atomic;
using std::condition_variable;
using std::lock_guard;
using std::mutex;
using std::queue;
using std::string;
using std::thread;
using std::unique_lock;
using std::vector;
using std::chrono::high_resolution_clock;


class CallData : public AbstractCallData
{

public:
  enum ServiceType
  {
    inference = 0,
    printStatistics = 1
  };

public:
  CallData(InferenceService::AsyncService *service, ServerCompletionQueue *cq, ServiceType s_type)
      : service_(service), cq_(cq), responder_(&ctx_), printStatisticsResponder_(&ctx_), s_type_(s_type), status_(CREATE)
  {
    Proceed(NULL);
  }

  void Proceed(void *threadpool);

  InferenceRequest getRequestData()
  {
    return request_;
  }

  void sendBack(float *data, int size)
  {
    reply_.mutable_data()->Add(data, data + size);
    // reply_.set_id(request_.id());
    status_ = FINISH;
    responder_.Finish(reply_, Status::OK, this);
  }

private:
  InferenceService::AsyncService *service_;
  ServerCompletionQueue *cq_;
  ServerContext ctx_;
  InferenceRequest request_;
  PrintStatisticsRequest printStatisticsRequest_;
  InferenceResponse reply_;
  PrintStatisticsResponse printStatisticsReply_;
  ServerAsyncResponseWriter<InferenceResponse> responder_;
  ServerAsyncResponseWriter<PrintStatisticsResponse> printStatisticsResponder_;
  ServiceType s_type_;
  enum CallStatus
  {
    CREATE,
    PROCESS,
    FINISH
  };
  CallStatus status_; // The current serving state.
};

class ServerImpl final
{
public:
  ~ServerImpl()
  {
    server_->Shutdown();
    // Always shutdown the completion queue after the server.
    cq_->Shutdown();
  }

  ServerImpl(int batch_size, int threadNum_, int wait) : modelManager_(batch_size, threadNum_, wait)
  {
  }

  void Run();

private:
  void HandleRpcs(int i);

  std::shared_ptr<ServerCompletionQueue> cq_;
  InferenceService::AsyncService service_;
  OnnxMlirModelManager modelManager_;
  std::shared_ptr<Server> server_;
  vector<std::thread> async_threads;
  mutex mtx_;
};

#endif
