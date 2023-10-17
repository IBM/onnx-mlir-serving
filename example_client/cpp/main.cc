#include "grpc_client.h"

// ./main <path include test_data_set_*> <model_name>
int main(int argc, char** argv) {

  Dataset ds(argv[1], argv[2]);
  const char* address = "localhost:50051";
  if(argc > 3)
    address = argv[3];
  InferenceClient client(grpc::CreateChannel(address, grpc::InsecureChannelCredentials()));
  std::vector<float> out = client.Inference(ds.getInput(0));
  std::cout << "result size: " << out.size() << std::endl;
  for(size_t i = 0; i< out.size(); i++)
    std::cout << out[i] << std::endl;

  return 0;
}
