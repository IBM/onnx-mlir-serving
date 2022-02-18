#include "AIU_async_server.h"


void std::CallData::Proceed(void *threadpool){
  if (status_ == CREATE) {
    // Make this instance progress to the PROCESS state.
    status_ = PROCESS;
    switch (s_type_){
      case std::CallData::inference:
        service_->RequestInference(&ctx_, &request_, &responder_, cq_, cq_, this);
        break;
      case std::CallData::end:
        service_->RequestEnd(&ctx_, &endRequest_, &endResponder_, cq_, cq_, this);
        break;
      default:
        break;  
    }
    // service_->RequestInference(&ctx_, &request_, &responder_, cq_, cq_,this);
  } else if (status_ == PROCESS) {

    switch (s_type_){
      case std::CallData::inference:
        new std::CallData(service_, cq_,std::CallData::inference);
        static_cast<std::AIUThreadpool*>(threadpool)->addCallData(this);
        now = std::chrono::high_resolution_clock::now();
        break;
      case std::CallData::end:
        new std::CallData(service_, cq_,std::CallData::end);
        static_cast<std::AIUThreadpool*>(threadpool)->printlogs();
        status_ = FINISH;
        endResponder_.Finish(endReply_, Status::OK, this);
        break;
      default:
        break;  
    }
    // new std::CallData(service_, cq_);

    // // The actual processing.
    // // string prefix("Hello ");
    // // reply_.set_message(prefix + request_.data());

    // static_cast<std::AIUThreadpool*>(threadpool)->addCallData(this);
    // now = std::chrono::high_resolution_clock::now();
    // // And we are done! Let the gRPC runtime know we've finished, using the
    // // memory address of this instance as the uniquely identifying tag for
    // // the event.
    // // status_ = FINISH;
    // // responder_.Finish(reply_, Status::OK, this);
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
    new std::CallData(&service_, cq_.get(),std::CallData::inference);
    new std::CallData(&service_, cq_.get(),std::CallData::end);
    HandleRpcs();

    
  }

 private:


  // This can be run in multiple threads if needed.
  void HandleRpcs() {
    // Spawn a new CallData instance to serve new clients.
    void* tag;  // uniquely identifies a request.
    bool ok;
    while (true) {
      GPR_ASSERT(cq_->Next(&tag, &ok));
      GPR_ASSERT(ok);
      static_cast<std::CallData*>(tag)->Proceed(&tpool);
    }
  }

  std::unique_ptr<ServerCompletionQueue> cq_;
  InferenceService::AsyncService service_;
  std::AIUThreadpool tpool;
  std::unique_ptr<Server> server_;
  std::vector<std::thread> async_threads;
  std::mutex  mtx_;
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
  ServerImpl server(threadNum, wait);
  server.Run();

  return 0;
}
