# Inference mnist model with onnx-mlir-serving

## Steps

1. Build onnx-mlir-serving
   To build onnx-mlir-serving, you can use the provided Dockerfile. Run the following command to build the Docker image:
   
   ```shell
   > sudo docker build -t ibm/onnx-mlir-serving .
   ```

2. Get an ONNX model
   Obtain an ONNX model to use for inference. For example, you can download the mnist-12 model along with its input data sets from the [ONNX Model Zoo](https://github.com/onnx/models/). Use the following commands to download and extract the model:
   
   ```shell
   > wget https://github.com/onnx/models/raw/main/vision/classification/mnist/model/mnist-12.tar.gz
   > tar xvzf mnist-12.tar.gz
   ```
   This will create the `mnist-12` directory containing the model and its associated data sets.

3. Compile ONNX model with ONNX-MLIR

   The Docker image (`ibm/onnx-mlir-serving`) already includes the ONNX-MLIR compiler.

   Start a Docker container using the `ibm/onnx-mlir-serving` image and open a shell inside the container with the following command:

   ```shell
   > sudo docker run -it --name serving ibm/onnx-mlir-serving
   Emulate Docker CLI using podman. Create /etc/containers/nodocker to quiet msg.
   root@1ea8cff25c0d:/workdir#
   ```

   In another shell session, copy the `mnist-12` directory from the host to the running container using the following command:

   ```shell
   sudo docker cp mnist-12/ serving:/workdir
   ```

   Return to the shell environment of the container and navigate to the `mnist-12` directory. To compile the `mnist-12.onnx` file into executable code, run the following command:

   ```shell
   root@1ea8cff25c0d:/workdir/mnist-12# onnx-mlir -O3 --maccel=NNPA mnist-12.onnx
   ```

   After compilation, a new shared library file named `mnist-12.so` will be generated.

   ```shell
   root@1ea8cff25c0d:/workdir/mnist-12# ls
   mnist-12.onnx  mnist-12.so  test_data_set_0
   ```
   