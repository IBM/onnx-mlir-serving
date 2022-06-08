#include "AIU_async_server.h"
#include <ctime>

std::chrono::high_resolution_clock::time_point  originTime(std::chrono::seconds(1646319840));
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
        std::cout << "get printStatistics request " <<std::endl;
        static_cast<DLCModelManager*>(modelManager)->PrintLogs();
        std::cout << "get printStatistics request2 " <<std::endl;
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


// void print_dim(const ::onnx::TensorShapeProto_Dimension &dim)
// {
//   switch (dim.value_case())
//   {
//   case onnx::TensorShapeProto_Dimension::ValueCase::kDimParam:
//     std::cout << dim.dim_param();
//     break;
//   case onnx::TensorShapeProto_Dimension::ValueCase::kDimValue:
//     std::cout << dim.dim_value();
//     break;
//   default:
//     assert(false && "should never happen");
//   }
// }

// void print_io_info(const ::google::protobuf::RepeatedPtrField< ::onnx::ValueInfoProto > &info)
// {
//   for (auto input_data: info)
//   {
//     auto shape = input_data.type().tensor_type().shape();
//     std::cout << "  " << input_data.name() << ":";
//     std::cout << "[";
//     if (shape.dim_size() != 0)
//     {
//       int size = shape.dim_size();
//       for (int i = 0; i < size - 1; ++i)
//       {
//         print_dim(shape.dim(i));
//         std::cout << ",";
//       }
//       print_dim(shape.dim(size - 1));
//     }
//     std::cout << "]\n";
//   }
// }

// void try_onnx_model(){
//   std::ifstream input("model.onnx",std::ios::ate | std::ios::binary); // open file and move current position in file to the end
//   std::streamsize size = input.tellg(); // get current position in file
//   input.seekg(0,std::ios::beg); // move to start of file
//   std::vector<char> buffer(size);
//   input.read(buffer.data(),size); // read raw data
//   onnx::ModelProto model;
//   model.ParseFromArray(buffer.data(),size); // parse protobuf
//   auto graph = model.graph();
//   std::cout << "graph inputs:\n";
//   print_io_info(graph.input());
//   std::cout << "graph outputs:\n";
//   print_io_info(graph.output());

// }

int main(int argc, char** argv) {
  // std::AIUThreadPool tpool(5);
  int wait = 0;
  int threadNum = 10;
  int batch_size = 10;
  if (argc >= 2) {
    wait = std::stoi(argv[1]);
  }
  if (argc >= 3) {
    batch_size = std::stoi(argv[2]);
  }
  if(argc >=4){
    threadNum = std::stoi(argv[3]);
  }

  // try_onnx_model();
  // modelLoder.LoadModel("./library.so");
  ServerImpl server(batch_size, threadNum, wait);
  server.Run();

  return 0;
}
