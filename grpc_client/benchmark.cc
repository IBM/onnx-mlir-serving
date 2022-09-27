
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "utils/grpc_client.h"

using std::chrono::high_resolution_clock;


struct Timer {
  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
};

class MultiClientSimlate{

  double _recordStart;
  double _recordEnd; 
  double _totalTime;
  std::vector<std::thread> _client_threads;
  std::mutex _log_mutex;
  Dataset &ds;
  char _host[200];
  std::ofstream ofs;

  public:
    MultiClientSimlate(const char* host, Dataset &ds, int threadNum, double recordStart, double recordEnd, double totalTime, const char* logPrefix):ds(ds){

      _recordStart = recordStart*1000;
      _recordEnd = recordEnd*1000; 
      _totalTime = totalTime*1000;

      // char* _host = host;
      memcpy(_host, host, strlen(host));

      char logname[200]; 
      sprintf(logname, "build/log_%s.txt", logPrefix);

      ofs.open(logname, std::ios::out);

      createThread(threadNum);
      for (std::thread& thread : _client_threads) {
          //thread.detach();
          if(thread.joinable())
              thread.join();
      }

      ofs.close();

    }

    void createThread(int threadNum){
      high_resolution_clock::time_point startTime = high_resolution_clock::now();
      for(int64_t i = 0; i<threadNum; i++){
        
        _client_threads.emplace_back([this](high_resolution_clock::time_point startTime, size_t threadIndex){
          InferenceClient client(grpc::CreateChannel(_host, grpc::InsecureChannelCredentials()));

          // std::cout << "Using " << _host << std::endl;

          size_t imageSize = ds.getImageCount();
          //ds.shape, ds.rank
          int count = 0;
          bool isStartRecord = false;
          bool isEndRecord = false;
          bool isRun = false;
          int totalcount = 0;
          std::vector<Timer> timerRecord;
          while(true){
            int index = count % imageSize;

            if(!isStartRecord){
              // warn up
              double d = std::chrono::duration<double, std::milli>(high_resolution_clock::now() - startTime).count();
              if(d > _recordStart){
                isStartRecord = true;
              }
                
            }else{
              // check end time
              double d = std::chrono::duration<double, std::milli>(high_resolution_clock::now() - startTime).count();
              if(d > _recordEnd){
                isEndRecord = true;
              }
                
            }
              
            if(!isRun){
              // wait 5s to make sure all thread are ready
              double d = std::chrono::duration<double, std::milli>(high_resolution_clock::now() - startTime).count();
              if(d > 5000){
                isRun = true;
              }else{
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
              }
            }

            if(isRun){
              Timer t;
              t.start = high_resolution_clock::now();
              std::vector<float> out = client.Inference(ds.getImageData(index), ds.shape, ds.rank, ds.modelName);
              t.end = high_resolution_clock::now();
              if(isStartRecord && !isEndRecord){
                timerRecord.push_back(t);
              }
              count ++;
              totalcount ++;
            }

              
            if(isEndRecord){
              double d = std::chrono::duration<double, std::milli>(high_resolution_clock::now() - startTime).count();
              if( d > _totalTime ){
                for(Timer t:timerRecord){
                  std::lock_guard<std::mutex> log_lock(_log_mutex);
                  ofs << threadIndex << " ";
                  ofs << std::to_string(std::chrono::duration<double, std::nano>(t.end -t.start).count()) << std::endl;
                }
                break;
              }
            }

            
          }
        },startTime, i);
      }
    }
};

// ./Benchmark <input dir> <log_prefix> <client_thread_number>
int main(int argc, char** argv) {

  Dataset ds(argv[1]);
  int threadNum = 64;
  const char *logPrefix = "out1";
  if(argc > 2){
    logPrefix = argv[2];
  }

  if(argc > 3){
    threadNum = std::stoi(argv[3]);
  }


  const char* host = "localhost:50051";
  if((host = getenv("AIU_server"))){
    std::cout << "Using " << host << std::endl;
  }else{
    host = "localhost:50051";
    std::cout << "Using " << host << std::endl;
  }

  std::cout << threadNum << std::endl;  
  MultiClientSimlate s(host, ds, threadNum, 5, 20, 20, logPrefix);


  InferenceClient* client = new InferenceClient(grpc::CreateChannel(host, grpc::InsecureChannelCredentials()));
  client->printStatistics();


  return 0;
}
