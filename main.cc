
// #pragma GCC diagnostic ignored "-Wunused-parameter"
// #define _OPEN_THREADS 1
#include <future>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <map>
#include <stdio.h>
#include <string.h>
// #include <pthread.h>
#include <time.h>
#include "mlperf_sut.h"
#include <thread>
#include "date.h"


// #include "RtMemRef.h"
//
// extern "C" OrderedRtMemRefDict *
// _dyn_entry_point_main_graph(OrderedRtMemRefDict *);
using namespace date;

std::string ToString(mlperf::TestScenario scenario) {
  switch (scenario) {
    case mlperf::TestScenario::SingleStream:
      return "Single Stream";
    case mlperf::TestScenario::MultiStream:
      return "Multi Stream";
    case mlperf::TestScenario::MultiStreamFree:
      return "Multi Stream Free";
    case mlperf::TestScenario::Server:
      return "Server";
    case mlperf::TestScenario::Offline:
      return "Offline";
  }
//  assert(false);
  return "InvalidScenario";
}

std::string ToString(mlperf::TestMode mode) {
  switch (mode) {
    case mlperf::TestMode::SubmissionRun:
      return "Submission";
    case mlperf::TestMode::AccuracyOnly:
      return "Accuracy";
    case mlperf::TestMode::PerformanceOnly:
      return "Performance";
    case mlperf::TestMode::FindPeakPerformance:
      return "Find Peak Performance";
  }
//  assert(false);
  return "InvalidMode";
}

mlperf::TestMode parseMode(const char* mode){
  static std::map<std::string,mlperf::TestMode> const modeTable = {
    {"SubmissionRun",mlperf::TestMode::SubmissionRun},
    {"AccuracyOnly",mlperf::TestMode::AccuracyOnly},
    {"PerformanceOnly",mlperf::TestMode::PerformanceOnly},
    {"FindPeakPerformance",mlperf::TestMode::FindPeakPerformance}
  };
  auto it = modeTable.find(mode);
  if (it != modeTable.end()) {
    return it->second;
  } else {
    return mlperf::TestMode::PerformanceOnly;
  }
}

mlperf::TestScenario parseScenario(const char* scenario){
  static std::map<std::string,mlperf::TestScenario> const scenarioTable = {
    {"SingleStream",mlperf::TestScenario::SingleStream},
    {"MultiStream",mlperf::TestScenario::MultiStream},
    {"Server",mlperf::TestScenario::Server},
    {"Offline",mlperf::TestScenario::Offline}
  };
  auto it = scenarioTable.find(scenario);
  if (it != scenarioTable.end()) {
    return it->second;
  } else {
    return mlperf::TestScenario::SingleStream;
  }
}

void getModelName(const char* input, char* str){
  char buffer[20];
  strcpy(buffer, input);
  char *ptr;
  ptr = strtok(buffer, "/");
  char *pre_ptr;
  while (ptr != NULL){
    pre_ptr = ptr;
    ptr = strtok (NULL, "/");
  }

  ptr = strtok(pre_ptr, "_");
  // std::cout << ptr << std::endl;
  // std::cout << pre_ptr << std::endl;

  strcpy(str, ptr);
  // std::cout << str << std::endl;
}

// ./app aiuConfig.cfg PerformanceOnly SingleStream /aivol/input_data/resnetduc
int main(int argc, char* argv[]) {

  // time_t my_time = time(NULL);
  // printf("%s", ctime(&my_time));
  std::cout << std::chrono::system_clock::now() << std::endl;

  mlperf::TestSettings ts;
  ts.mode = parseMode(argv[2]);
  ts.scenario = parseScenario(argv[3]);
  char model[10]; 
  getModelName(argv[4],model);
  ts.FromConfig(argv[1], model, argv[3]);
  if (!strncmp("mnist", model, 5)){
    ts.single_stream_expected_latency_ns = 280000;
    std::cout <<"mnist "<<ts.single_stream_expected_latency_ns << std::endl;
  }
  // std::cout << ToString(ts.scenario) <<" "<< ToString(ts.mode) << std::endl;
  // std::cout <<ts.multi_stream_samples_per_query <<" "<< ts.min_duration_ms << std::endl;

  // grpc_tests::InferenceServiceClient client(grpc::CreateChannel("10.10.170.2:50051", grpc::InsecureChannelCredentials())); 
  // client.start();
  grpc_tests::InferenceService2Client client(grpc::CreateChannel("10.10.170.2:50051", grpc::InsecureChannelCredentials()));
  grpc_tests::Dataset ds(argv[4], client);
  grpc_tests::QuerySampleLibraryNull qsl(ds);
  mlperf::LogSettings log_settings;
  log_settings.log_output.copy_summary_to_stdout = true;
  int threadNum = std::thread::hardware_concurrency();

  

  switch (ts.scenario) {
    case mlperf::TestScenario::SingleStream:{
      grpc_tests::SystemUnderTestSingleStream sut(ds);
      mlperf::StartTest(&sut, &qsl, ts, log_settings);
      break;
    }
    case mlperf::TestScenario::MultiStream:{
      int numT = threadNum;
      if(ts.multi_stream_max_async_queries > numT)
        numT = ts.multi_stream_max_async_queries;
      grpc_tests::SystemUnderTestPool sut(ds, numT);
      mlperf::StartTest(&sut, &qsl, ts, log_settings);
      break;
    }
    case mlperf::TestScenario::Server:{
      // grpc_tests::SystemUnderTestStdAsync sut(ds);
      grpc_tests::SystemUnderTestPool sut(ds, threadNum);
      mlperf::StartTest(&sut, &qsl, ts, log_settings);
      break;
    }
    case mlperf::TestScenario::Offline:{
      grpc_tests::SystemUnderTestPool sut(ds, threadNum);
      mlperf::StartTest(&sut, &qsl, ts, log_settings);
      break;
    }
    default:{
      std::cout << "err" << std::endl;
      break;
    }
  }
  // client.end();
  // my_time = time(NULL);
  // printf("%s", ctime(&my_time));
  client.end();
  std::cout << std::chrono::system_clock::now() << std::endl;
  return 0;
}
