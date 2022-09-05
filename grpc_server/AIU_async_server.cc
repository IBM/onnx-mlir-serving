#pragma GCC diagnostic ignored "-Wdelete-non-virtual-dtor"
#include "aiu_async_server.h"
#include <ctime>

std::chrono::high_resolution_clock::time_point  originTime = std::chrono::high_resolution_clock::now();
DLCModelLoader modelLoder;


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
        static_cast<DLCModelManager*>(modelManager)->AddModel(this);
        now = high_resolution_clock::now();
        break;
      case CallData::printStatistics:
        new CallData(service_, cq_,CallData::printStatistics);
        static_cast<DLCModelManager*>(modelManager)->PrintLogs();
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


int main(int argc, char** argv) {
  // std::AIUThreadPool tpool(5);
  int wait = 0; 
  int batch_size = 1;
  int threadNum = 1;

  int argIndex = 1;

  while(argIndex < argc){
    char *curArg = argv[argIndex];
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
    if(argIndex == 1){
      wait = std::stoi(argv[1]);
      argIndex ++;
      continue;
    }
    if(argIndex == 2){
      batch_size = std::stoi(argv[2]);
      argIndex ++;
      continue;
    }       
    if(argIndex == 3){
      threadNum = std::stoi(argv[3]);
      argIndex ++;
      continue;
    }          
  }


  std::cout << "wait time " << wait << " ns" << std::endl;
  std::cout << "batch max size " << batch_size << std::endl;
  std::cout << "thread number " << threadNum << std::endl;
  ServerImpl server(batch_size, threadNum, wait);
  server.Run();

  return 0;
}
