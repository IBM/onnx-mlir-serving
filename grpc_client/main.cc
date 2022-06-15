#include "utils/grpc_client.h"

// ./main <dataset_path> 
int main(int argc, char** argv) {

  Dataset ds(argv[1]);
  InferenceClient client(grpc::CreateChannel("localhost:50051", grpc::InsecureChannelCredentials()));
  std::vector<float> out = client.Inference(ds.getImageData(0), ds.shape, ds.rank, ds.modelName);
  for(size_t i = 0; i< out.size(); i++)
    std::cout << out[i] << std::endl;

  return 0;
}
