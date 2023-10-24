# Inference MNIST Model with ONNX-MLIR-Serving

## Tutorial

This tutorial demonstrates how to perform inference on the MNIST model using the ONNX-MLIR-Serving framework. Follow the steps below to set up the environment and perform the inference.

### Step 1: Build ONNX-MLIR-Serving

1. Start by building the ONNX-MLIR-Serving Docker image. Navigate to the `onnx-mlir-serving` directory and run the following command:
   
   ```shell
   cd onnx-mlir-serving
   sudo docker build -t onnx/onnx-mlir-server .
   ```

### Step 2: Obtain the MNIST ONNX Model

1. Download the MNIST model and its associated input data sets from the ONNX Model Zoo. Use the following commands to download and extract the model:
   
   ```shell
   wget https://github.com/onnx/models/raw/main/vision/classification/mnist/model/mnist-12.tar.gz
   tar xvzf mnist-12.tar.gz
   ```
   This will create the `mnist-12` directory, which contains the model and its associated data sets.

### Step 3: Compile the ONNX Model with ONNX-MLIR

1. Start a Docker container using the `onnx/onnx-mlir-server` image. Use the following command to run the container and open a shell inside it:
   (The prompt "container#" denotes the shell inside the container)
   
   ```shell
   sudo docker run -it --name serving onnx/onnx-mlir-server
   Emulate Docker CLI using podman. Create /etc/containers/nodocker to quiet msg.
   container#
   ```

2. In another shell session, copy the `mnist-12` directory from the host to the running container using the following command:
   
   ```shell
   sudo docker cp mnist-12/ serving:/workdir
   ```

3. Return to the shell environment of the container and navigate to the `mnist-12` directory. Use the following command to compile the `mnist-12.onnx` model into executable code:
   
   ```shell
   container# cd mnist-12/
   container# onnx-mlir -O3 --maccel=NNPA mnist-12.onnx
   ```

   This will generate a shared library file named `mnist-12.so` in the `mnist-12` directory.

4. Verify that the files were generated successfully:
   
   ```shell
   container# ls
   mnist-12.onnx  mnist-12.so  test_data_set_0
   ```

### Step 4: Configure the Model for GRPC Serving

1. Prepare a directory structure for the model inside the `models` directory of the `grpc_server`:
   
   ```
   grpc_server
   ├── models
   │   └── mnist
   │       ├── config
   │       ├── model.so
   │       └── model.onnx
   ```

2. Copy the model's ONNX and compiled `.so` files to the `models/mnist` directory inside the container:
   
   ```shell
   container# cd /workdir/onnx-mlir-serving/cmake/build
   container# mkdir -p models/mnist
   container# cd models/mnist/
   container# cp /workdir/mnist-12/mnist-12.onnx model.onnx
   container# cp /workdir/mnist-12/mnist-12.so model.so
   ```

3. Generate a configuration file for the model based on its ONNX file:
   
   ```shell
   container# ../../utils/OnnxReader ./model.onnx
   container# ls
   config  model.onnx  model.so
   ```

   The `config` file should be in JSON format and contains information about the model's input, output, and maximum batch size.

### Step 5: Start the GRPC Server

1. Return to the `grpc_server` directory and start the server:
   
   ```shell
   container# cd /workdir/onnx-mlir-serving/cmake/build
   container# ./grpc_server
   wait time 0ns
   batch max size 1
   thread number 1
   Server listening on 0.0.0.0:50051
   ```

### Step 6: Send Inference Request with a Client

1. Start an client to send inference request for inference on the MNIST model. Use the provided C++ client example `Client` with the following command:
   
   ```shell
   container# cd /workdir/onnx-mlir-serving/cmake/build/cpp
   container# ./Client /workdir/mnist-12 mnist
   ```

   The output will be an array representing the possibilities of each digit (0-9) based on the input. The inference result will be the digit with the highest possibility.

   Example output:
   
   ```
   result size: 10
   -48.8125
   -4.6875
   -21.9062
   -13.3906
   75.25
   -8.10938
   -52.3125
   22.625
   -16.6875
   48.0625
   ```

   In this example, the highest possibility is 75.25 for the 5th number (digit 4), indicating that digit 4 is the predicted result.

Note: Please make sure to adjust the paths and commands as necessary based on your specific setup and environment.
   
   This program sends first record of mnist dataset (input*.pb) to get the inference result.

   If you like, you can update example client code to read other datasets or your own record.

#### Client Example Code

```C
#include "grpc_client.h"

// ./main <path include test_data_set_*> <model_name>
int main(int argc, char** argv) {

  Dataset ds(argv[1], argv[2]);
  const char* address = "localhost:50051";
  if(argc > 3)
    address = argv[3];
  InferenceClient client(grpc::CreateChannel(address, grpc::InsecureChannelCredentials()));
  // ds.getInput(0) just get first record only.
  std::vector<float> out = client.Inference(ds.getInput(0));
  std::cout << "result size: " << out.size() << std::endl;
  for(size_t i = 0; i< out.size(); i++)
    std::cout << out[i] << std::endl;

  return 0;
}
```
