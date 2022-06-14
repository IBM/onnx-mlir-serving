#include <limits.h>
#include <algorithm>
#include "gtest/gtest.h"
#include "utils/GRpcClient.h"
#include <chrono>
#include <thread>
using namespace std::chrono_literals;
namespace {

class ServerTest : public testing::Test {
 protected:
    // Remember that SetUp() is run immediately before a test starts.
    // This is a good place to record the start time.
    void SetUp() override { 
      std::cout << "Start aiu server with simplest way" << std::endl;
      system("bash -c './AIU_async_server 0 1 1' &");
      std::this_thread::sleep_for(2s);
      
    }
    // TearDown() is invoked immediately after a test finishes.  Here we
    // check if the test was too slow.
    void TearDown() override {
      std::cout << "Stop aiu server" << std::endl;
      auto a = system("pkill -e AIU");
    } 
      

      // Gets the time when the test finishes

  };




  TEST_F(ServerTest, mnist0) {
    system("wait 5");
    Dataset ds("./models/mnist");
    InferenceClient client(grpc::CreateChannel("localhost:50051", grpc::InsecureChannelCredentials()));
    std::vector<float> out_vector = client.Inference(&ds, 0);
    for (auto value:out_vector)
        std::cout << value << std::endl;
    auto maxPosition = max_element(out_vector.begin(),out_vector.end()) - out_vector.begin(); 
    EXPECT_EQ(4, maxPosition);
    out_vector = client.Inference(&ds, 0);
    for (auto value:out_vector)
        std::cout << value << std::endl;
    maxPosition = max_element(out_vector.begin(),out_vector.end()) - out_vector.begin(); 
    EXPECT_EQ(4, maxPosition);
    // auto out_vector2 = client.Inference(&ds, 1);
    // auto maxPosition2 = max_element(out_vector2.begin(),out_vector2.end()) - out_vector2.begin(); 
    // EXPECT_EQ(4, maxPosition2);
  }
} // namespace