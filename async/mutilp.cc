#include "sut.h"

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

  // Configure the test settings
  mlperf::TestSettings testSettings;
  testSettings.scenario = mlperf::TestScenario::MultiStream;
  testSettings.mode = mlperf::TestMode::PerformanceOnly;
  testSettings.multi_stream_samples_per_query = multi_stream_samples_per_query; //10000000;  // 10ms
  testSettings.multi_stream_target_qps = 10;
  testSettings.multi_stream_target_latency_ns = 4000000;
  testSettings.multi_stream_target_latency_percentile = 0.99;
  testSettings.min_duration_ms = 60000;
  testSettings.min_query_count = 10;


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
  logSettings.enable_trace = true;


  std::cout << "Using QueueSUT" << std::endl;
  sut.reset(new QueueSUT(1, ds));

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


  InferenceClient* client = new InferenceClient(grpc::CreateChannel( "localhost:50051", grpc::InsecureChannelCredentials()));
  client->end();
  std::cout << "send end" << std::endl;

  return 0;
}
