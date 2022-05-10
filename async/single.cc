#include "sut.h"

// ./single /aivol/inputs/ccf1_inputs 0 1
int main(int argc, char** argv) {
  assert(argc >= 3 && "Need to pass input dir and log suffix");


  bool grpcCall{true};
  if (argc >= 4) {
    grpcCall = std::stoi(argv[3]) != 0;
  }


  Dataset ds(argv[1],grpcCall);


  QSL qsl(ds);
  std::unique_ptr<mlperf::SystemUnderTest> sut;

  // Configure the test settings
  mlperf::TestSettings testSettings;
  testSettings.scenario = mlperf::TestScenario::SingleStream;
  testSettings.mode = mlperf::TestMode::PerformanceOnly;
  testSettings.single_stream_expected_latency_ns = 300000; //10000000;  // 10ms
  testSettings.single_stream_target_latency_percentile = 0.99;
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


  std::cout << "Using BasicSUT" << std::endl;
  sut.reset(new BasicSUT(ds));

  // Start test
  std::cout << "Start test..." << std::endl;
  mlperf::StartTest(sut.get(), &qsl, testSettings, logSettings);
  std::cout << "Test done. Clean up SUT..." << std::endl;
  sut.reset();
  std::cout << "Done!" << std::endl;
  if(std::stoi(argv[2]) == 0){
    std::cout << "send end" << std::endl;
    InferenceClient* client = new InferenceClient(grpc::CreateChannel( "localhost:50051", grpc::InsecureChannelCredentials()));
    client->end();
  }

  // std::cout << "send end" << std::endl;
  // InferenceClient* client = new InferenceClient(grpc::CreateChannel( "localhost:50051", grpc::InsecureChannelCredentials()));
  // client->end();

  return 0;
}
