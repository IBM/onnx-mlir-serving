#include "sut.h"

int main(int argc, char** argv) {
  assert(argc >= 2 && "Need to pass input dir");

  bool useQueue{false};
  int maxSize{1};
  bool server_coalesce_queries{true};
  int num_issue_threads{0};
  bool grpcCall{true};
  int target_qps{1};
  if (argc >= 3) {
    grpcCall = std::stoi(argv[2]) != 0;
  }
  if (argc >= 4) {
    target_qps = std::stoi(argv[3]);
    std::cout << "target_qps = " << target_qps << std::endl;
  }
  if (argc >= 5) {
    useQueue = std::stoi(argv[4]) != 0;
  }
  if (argc >= 6) {
    num_issue_threads = std::stoi(argv[5]);
  }

  Dataset ds(argv[1],grpcCall);


  QSL qsl(ds);
  std::unique_ptr<mlperf::SystemUnderTest> sut;

  // Configure the test settings
  mlperf::TestSettings testSettings;
  testSettings.scenario = mlperf::TestScenario::Server;
  testSettings.mode = mlperf::TestMode::PerformanceOnly;
  testSettings.server_target_qps = target_qps;
  testSettings.server_target_latency_ns = 300000; //10000000;  // 10ms
  testSettings.server_target_latency_percentile = 0.99;
  testSettings.min_duration_ms = 60000;
  testSettings.min_query_count = 10;
  testSettings.server_max_async_queries = -1;
  testSettings.server_coalesce_queries = server_coalesce_queries;
  std::cout << "testSettings.server_coalesce_queries = "
            << (server_coalesce_queries ? "True" : "False") << std::endl;
  // testSettings.server_num_issue_query_threads = num_issue_threads;
  std::cout << "num_issue_threads = " << num_issue_threads << std::endl;

  // Configure the logging settings
  mlperf::LogSettings logSettings;
  logSettings.log_output.outdir = "build";
  logSettings.log_output.prefix = "mlperf_log_";
  logSettings.log_output.suffix = "";
  logSettings.log_output.prefix_with_datetime = false;
  logSettings.log_output.copy_detail_to_stdout = false;
  logSettings.log_output.copy_summary_to_stdout = true;
  logSettings.log_mode = mlperf::LoggingMode::AsyncPoll;
  logSettings.log_mode_async_poll_interval_ms = 1000;
  logSettings.enable_trace = true;

  // Choose SUT
  if (num_issue_threads == 0) {
    if (useQueue) {
      std::cout << "Using QueueSUT with " << num_issue_threads
                << " complete threads" << std::endl;
      sut.reset(new QueueSUT(num_issue_threads, ds));
    } else {
      testSettings.mode = mlperf::TestMode::AccuracyOnly;
      std::cout << "Using BasicSUT" << std::endl;
      sut.reset(new BasicSUT(ds));
    }
  } else {
    if (useQueue) {
      std::cout << "Using QueueSUT with " << num_issue_threads
                << " complete threads" << std::endl;
      // std::cerr << "!!!! MultiQueueSUT is NOT implemented yet !!!!"
      //           << std::endl;
      sut.reset(new QueueSUT(num_issue_threads, ds));
      // sut.reset(new MultiQueueSUT(num_issue_threads, numCompleteThreads,
      // maxSize));
    } else {
      std::cout << "Using MultiBasicSUT" << std::endl;
      testSettings.server_num_issue_query_threads = num_issue_threads;
      sut.reset(new MultiBasicSUT(num_issue_threads, ds));
    }
  }

  // Start test
  std::cout << "Start test..." << std::endl;
  mlperf::StartTest(sut.get(), &qsl, testSettings, logSettings);
  std::cout << "Test done. Clean up SUT..." << std::endl;
  sut.reset();
  std::cout << "Done!" << std::endl;
  InferenceClient* client = new InferenceClient(grpc::CreateChannel( "localhost:50051", grpc::InsecureChannelCredentials()));
  client->end();
  return 0;
}
