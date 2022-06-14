#include <limits.h>

#include "gtest/gtest.h"
#include "utils/GRpcClient.h"
namespace {




TEST(mnist, 1) {

  Dataset ds("./models/mnist");
  InferenceClient client(grpc::CreateChannel("localhost:50051", grpc::InsecureChannelCredentials()));
  std::vector<float> out = client.Inference(&ds, 0);
  std::cout << out[0] << std::endl;

}
} // namespace