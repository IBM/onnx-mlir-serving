// #pragma GCC diagnostic ignored "-Wdelete-non-virtual-dtor"
#include "grpc_server.h"
#include <ctime>

std::chrono::high_resolution_clock::time_point  originTime = std::chrono::high_resolution_clock::now();
OnnxMlirModelLoader modelLoder;


void CallData::Proceed(void *modelManager){
  if (status_ == CREATE) {
    // Make this instance progress to the PROCESS state.
    status_ = PROCESS;
    switch (s_type_){
      case CallData::inference:
        service_->RequestInference(&ctx_, &request_, &responder_, cq_, cq_, this);
        break;
      case CallData::printStatistics:
        service_->RequestPrintStatistics(&ctx_, &printStatisticsRequest_, &printStatisticsResponder_, cq_, cq_, this);
        break;
      default:
        break;
    }
    // service_->RequestInference(&ctx_, &request_, &responder_, cq_, cq_,this);
  } else if (status_ == PROCESS) {

    switch (s_type_){
      case CallData::inference:
        new CallData(service_, cq_,CallData::inference);
        static_cast<OnnxMlirModelManager*>(modelManager)->AddModel(this);
        now = high_resolution_clock::now();
        break;
      case CallData::printStatistics:
        new CallData(service_, cq_,CallData::printStatistics);
        static_cast<OnnxMlirModelManager*>(modelManager)->PrintLogs();
        status_ = FINISH;
        printStatisticsResponder_.Finish(printStatisticsReply_, Status::OK, this);
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

void ServerImpl::Run(){
    std::string server_address("0.0.0.0:50051");
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service_);
    cq_ = builder.AddCompletionQueue();
    server_ = builder.BuildAndStart();
    std::cout << "Server listening on " << server_address << std::endl;

    new CallData(&service_, cq_.get(),CallData::printStatistics);
    new CallData(&service_, cq_.get(),CallData::inference);
    HandleRpcs(0);
    // for(int i = 0; i < 2; i++){
    //   async_threads.emplace_back([this](int i){
    //     HandleRpcs(i);
    //   },i);
    // }
}

void ServerImpl::HandleRpcs(int i){
    // Spawn a new CallData instance to serve new clients.

    void* tag;  // uniquely identifies a request.
    bool ok;
    while (true) {
      
      lock_guard<mutex> lock(mtx_);
      GPR_ASSERT(cq_->Next(&tag, &ok));
      
      // GPR_ASSERT(ok);
      if(ok){
        static_cast<CallData*>(tag)->Proceed(&modelManager_);
      }

    }  
}

int printHelp(){
  std::cout << "usage: server [options]" << std::endl;
  std::cout << "-w arg     " << "wait time for batch size" << std::endl;
  std::cout << "-b arg     " << "server side batch size" << std::endl;
  std::cout << "-n arg     " << "thread number" << std::endl;
}

int main(int argc, char** argv) {
  // std::AIInfrenceThreadPool tpool(5);
  int wait = 0; 
  int batch_size = 1;
  int threadNum = 1;

  int argIndex = 1;

  while(argIndex < argc){
    char *curArg = argv[argIndex];
    if(strcmp(curArg, "-h") == 0){
      printHelp();
      return 0;
    }
    if(strcmp(curArg, "-w") == 0){
      wait = std::stoi(argv[argIndex + 1]);
      argIndex = argIndex + 2;
      continue;
    }
    if(strcmp(curArg, "-b") == 0){
      batch_size = std::stoi(argv[argIndex + 1]);
      argIndex = argIndex + 2;
      continue;
    }
    if(strcmp(curArg, "-n") == 0){
      threadNum = std::stoi(argv[argIndex + 1]);
      argIndex = argIndex + 2;
      continue;
    }
    printHelp();
    return 0;       
  }

  // std::cout << "wait time " << wait << " ns" << std::endl;
  // std::cout << "batch max size " << batch_size << std::endl;
  // std::cout << "thread number " << threadNum << std::endl;
  ServerImpl server(batch_size, threadNum, wait);
  server.Run();

  return 0;
}
