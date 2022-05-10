#include "sut.h"
#include<stdlib.h>
#include <time.h>

// m /aivol/inputs/ccf1_inputs <prefix> <multi_stream_samples_per_query> <grpcCall>
int main(int argc, char** argv) {
  assert(argc >= 3 && "Need to pass input dir and log suffix");

  int multi_stream_samples_per_query {10};
  bool grpcCall{true};
  if (argc >= 4) {
    multi_stream_samples_per_query = std::stoi(argv[3]);
  }
  if (argc >= 5) {
    grpcCall = std::stoi(argv[4]) != 0;
  }


  Dataset ds(argv[1],grpcCall);


  QSL qsl(ds);
  std::unique_ptr<mlperf::SystemUnderTest> sut;

  time_t my_time = time(NULL);
  printf("%s", ctime(&my_time));

  // Configure the test settings
  mlperf::TestSettings testSettings;

  testSettings.scenario = mlperf::TestScenario::MultiStream;
  testSettings.multi_stream_samples_per_query = multi_stream_samples_per_query; //10000000;  // 10ms
  testSettings.multi_stream_target_qps = 130;
  testSettings.multi_stream_target_latency_ns = 7679801;
  testSettings.multi_stream_target_latency_percentile = 0.99;


  // testSettings.scenario = mlperf::TestScenario::SingleStream;
  testSettings.single_stream_expected_latency_ns = 300000; //10000000;  // 10ms
  testSettings.single_stream_target_latency_percentile = 0.99;


  testSettings.min_duration_ms = 180000;
  testSettings.min_query_count = 10;
  testSettings.mode = mlperf::TestMode::PerformanceOnly;
  // Configure the logging settings

  mlperf::LogSettings logSettings;
  logSettings.log_output.outdir = "build";
  logSettings.log_output.prefix = "mlperf_log_";
  logSettings.log_output.suffix = argv[2];
  logSettings.log_output.prefix_with_datetime = false;
  logSettings.log_output.copy_detail_to_stdout = false;
  logSettings.log_output.copy_summary_to_stdout = false;
  logSettings.log_mode = mlperf::LoggingMode::AsyncPoll;
  logSettings.log_mode_async_poll_interval_ms = 1000;
  logSettings.enable_trace = false;



  std::cout << "Using QueueSUT" << std::endl;
  sut.reset(new BasicSUT(ds));

  // Start test
  std::cout << "Start test..." << std::endl;
  mlperf::StartTest(sut.get(), &qsl, testSettings, logSettings);
  std::cout << "Test done. Clean up SUT..." << std::endl;
  sut.reset();
  std::cout << "Done!" << std::endl;
//   if(std::stoi(argv[2]) == 0){
//     std::cout << "send end" << std::endl;
//     InferenceClient* client = new InferenceClient(grpc::CreateChannel( "localhost:50051", grpc::InsecureChannelCredentials()));
//     client->end();
//   }

  my_time = time(NULL);
  printf("%s", ctime(&my_time));

  InferenceClient* client = new InferenceClient(grpc::CreateChannel( "10.1.20.98:50051", grpc::InsecureChannelCredentials()));
  client->end();
  std::cout << "send end" << std::endl;



  return 0;
}
