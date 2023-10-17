#include <unordered_map>

#include "grpc_client.h"


class MultiClientSimlate{

  double _recordStart;
  double _recordEnd; 
  double _totalTime;
  double _actuTime;
  std::vector<std::thread> _client_threads;
  std::mutex _log_mutex;
  Dataset &ds;
  char _host[200];
  std::ofstream ofs;
  std::vector<double> latencys;
  std::unordered_map<int, std::vector<float>> _resultMap;

  public:
    MultiClientSimlate(char* host, Dataset &ds, int threadNum, double recordStart, double recordEnd, double totalTime, char* logPrefix):ds(ds){

      _recordStart = recordStart*1000;
      _recordEnd = recordEnd*1000; 
      _totalTime = totalTime*1000;

      // char* _host = host;
      memcpy(_host, host, strlen(host));

      char logname[200]; 
      sprintf(logname, "build/log_%s.txt", logPrefix);

      ofs.open(logname, std::ios::out);

      // std::unordered_map<std::string, std::vector<float>> resultMap;

      high_resolution_clock::time_point startTime = high_resolution_clock::now();
      createThread(threadNum);
      for (std::thread& thread : _client_threads) {
          //thread.detach();
          if(thread.joinable())
              thread.join();
      }
      high_resolution_clock::time_point endTime = high_resolution_clock::now();
      _actuTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();


      ofs.close();

    }

    void createThread(int threadNum){

      high_resolution_clock::time_point startTime = high_resolution_clock::now();
      for(size_t i = 0; i<threadNum; i++){

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
          std::map<int, std::vector<float>> resultMap;
          while(true){
            int index = count % imageSize;

            if(!isStartRecord){
              double d = std::chrono::duration<double, std::milli>(high_resolution_clock::now() - startTime).count();
              if(d > _recordStart){
                isStartRecord = true;
              }

            }else{
              double d = std::chrono::duration<double, std::milli>(high_resolution_clock::now() - startTime).count();
              if(d > _recordEnd){
                isEndRecord = true;
              }

            }

            if(!isRun){
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
              std::vector<float> out = client.Inference(ds.getInput(index));
              t.end = high_resolution_clock::now();
              resultMap.emplace(index,out);
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
                  double latency = std::chrono::duration<double, std::micro>(t.end -t.start).count();
                  latencys.emplace_back(latency);
                  ofs << std::to_string(latency) << std::endl;
                }

                for (auto iter = resultMap.begin(); iter != resultMap.end(); ++iter){
                  std::lock_guard<std::mutex> log_lock(_log_mutex);
                  _resultMap.emplace(iter->first, iter->second);
                }

                break;
              }
            }


          }
        },startTime, i);
      }
    }

    char Bin2Hex(uint8_t four_bits) {
      char number = '0' + four_bits;
      char letter = ('A' - 10) + four_bits;
      return four_bits < 10 ? number : letter;
    }

    const std::string ArgValueTransform(const std::vector<uint8_t>* data) {
      if (data == nullptr) {
        return "\"\"";
      }
      std::string hex;
      hex.reserve(data->size() + 2);
      hex.push_back('"');
      for (auto b : *data) {
        hex.push_back(Bin2Hex(b >> 4));
        hex.push_back(Bin2Hex(b & 0x0F));
      }
      hex.push_back('"');
      return hex;
    }

    double meanLatency(){
      double sum = std::accumulate(latencys.begin(), latencys.end(), 0.0);
      // std::cout << sum << " " << latencys.size() << std::endl;
      double mean = sum / latencys.size();

      std::cout << "total time: " << _actuTime << std::endl;
      std::cout << "qps: " << latencys.size()  / _actuTime * 1000 << std::endl;


      char logname[200];
      sprintf(logname, "build/log_acc%s.json", "2");
      std::ofstream ofs_acc;
      size_t float_size = sizeof(float);
      ofs_acc.open(logname, std::ios::out);
      ofs_acc << "[" <<  std::endl;
      for (auto iter = _resultMap.begin(); iter != _resultMap.end(); ++iter){

        uint8_t* src_begin = reinterpret_cast<uint8_t*>(iter->second.data());
        uint8_t* src_end = src_begin + iter->second.size()*float_size;
        std::vector<uint8_t>* data = new std::vector<uint8_t>(src_begin, src_end);

        ofs_acc << "{\"qsl_idx\":\"" <<  iter->first << "\",\"data\":"<< ArgValueTransform(data) << "}," <<  std::endl;
      }
      ofs_acc << "]";
      return mean;
    }
};

// ./Benchmark ccf1 /aivol/inputs/ccf1_inputs 1
// ./Benchmark model_name inputs threadNum 
int main(int argc, char** argv) {

  Dataset ds(argv[2], argv[1]);
  int threadNum = 64;
  char* logPrefix = argv[1];

  if(argc > 3){
    threadNum = std::stoi(argv[3]);
  }

  char* host = "localhost:50051";
  if((host = getenv("grpc_server")))
    std::cout << "Using " << host << std::endl;
  else
    std::cout << "Using " << host << std::endl;


  std::cout << "number of threads: " << threadNum << std::endl; 
  MultiClientSimlate s(host, ds, threadNum, 0, 60, 60, logPrefix);

  InferenceClient* client = new InferenceClient(grpc::CreateChannel(host, grpc::InsecureChannelCredentials()));
  client->printStatistics();

  double latency = s.meanLatency();
  std::cout << "mean latency: " << latency << std::endl;
  return 0;
}
