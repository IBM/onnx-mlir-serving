#include "AIU_async_server.h"
#include <ctime>

void CallData::Proceed(void *threadpool){
  if (status_ == CREATE) {
    // Make this instance progress to the PROCESS state.
    status_ = PROCESS;
    switch (s_type_){
      case CallData::inference:
        service_->RequestInference(&ctx_, &request_, &responder_, cq_, cq_, this);
        break;
      case CallData::end:
        service_->RequestEnd(&ctx_, &endRequest_, &endResponder_, cq_, cq_, this);
        break;
      default:
        break;
    }
    // service_->RequestInference(&ctx_, &request_, &responder_, cq_, cq_,this);
  } else if (status_ == PROCESS) {

    switch (s_type_){
      case CallData::inference:
        new CallData(service_, cq_,CallData::inference);
        static_cast<AIUThreadpool*>(threadpool)->addCallData(this);
        now = high_resolution_clock::now();
        break;
      case CallData::end:
        new CallData(service_, cq_,CallData::end);
        static_cast<AIUThreadpool*>(threadpool)->printlogs();
        status_ = FINISH;
        endResponder_.Finish(endReply_, Status::OK, this);
        break;
      default:
        break;
    }
  } else {
    GPR_ASSERT(status_ == FINISH);
    // Once in the FINISH state, deallocate ourselves (CallData).
    delete this;
  }
}

class ServerImpl final {
 public:
  ~ServerImpl() {
    server_->Shutdown();
    // Always shutdown the completion queue after the server.
    cq_->Shutdown();
  }

  ServerImpl(int threadNum_, int wait):tpool(threadNum_, wait){
    // tpool = std::AIUThreadpool(5);
  }

  // There is no shutdown handling in this code.
  void Run() {
    std::string server_address("0.0.0.0:50051");

    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    // Register "service_" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *asynchronous* service.
    builder.RegisterService(&service_);
    // Get hold of the completion queue used for the asynchronous communication
    // with the gRPC runtime.
    cq_ = builder.AddCompletionQueue();
    // Finally assemble the server.
    server_ = builder.BuildAndStart();
    std::cout << "Server listening on " << server_address << std::endl;

    // Proceed to the server's main loop.
    // new std::CallData(&service_, cq_.get(),std::CallData::inference);
    // new std::CallData(&service_, cq_.get(),std::CallData::end);
    // HandleRpcs();
    new CallData(&service_, cq_.get(),CallData::end);
    for(int i = 0; i < 20; i++){
      async_threads.emplace_back([this](int i){
        HandleRpcs(i);
      },i);
    }

    for (std::thread& thread : async_threads) {
        //thread.detach();
        if(thread.joinable())
            thread.join();
    }

  }

 private:


  // This can be run in multiple threads if needed.
  void HandleRpcs(int i) {
    // Spawn a new CallData instance to serve new clients.

    new CallData(&service_, cq_.get(),CallData::inference);

    void* tag;  // uniquely identifies a request.
    bool ok;
    while (true) {
      {
      lock_guard<mutex> lock(mtx_);
      GPR_ASSERT(cq_->Next(&tag, &ok));
      }
      // GPR_ASSERT(ok);
      if(ok){
        static_cast<CallData*>(tag)->Proceed(&tpool);
      }

    }
  }

  std::shared_ptr<ServerCompletionQueue> cq_;
  InferenceService::AsyncService service_;
  AIUThreadpool tpool;
  std::shared_ptr<Server> server_;
  vector<std::thread> async_threads;
  mutex  mtx_;
};

int main(int argc, char** argv) {
  // std::AIUThreadpool tpool(5);
  int wait = 0;
  int threadNum = 10;
  if (argc >= 2) {
    wait = std::stoi(argv[1]);
  }
  if (argc >= 3) {
    threadNum = std::stoi(argv[2]);
  }

  modelLoder.LoadModel("./library.so");

  ServerImpl server(threadNum, wait);
  server.Run();

  return 0;
}
