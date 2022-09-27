#include "utils/grpc_client.h"

// ./main <dataset_path> 
int main(int argc, char** argv) {

  Dataset ds(argv[1]);
  const char* address = "localhost:50051";
  if(argc > 2)
    address = argv[2];
  InferenceClient client(grpc::CreateChannel(address, grpc::InsecureChannelCredentials()));
  std::vector<float> out = client.Inference(ds.getImageData(0), ds.shape, ds.rank, ds.modelName);
  std::cout << "result size: " << out.size() << std::endl;
  for(size_t i = 0; i< out.size(); i++)
    std::cout << out[i] << std::endl;

  return 0;
}
