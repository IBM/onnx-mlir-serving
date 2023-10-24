<meta charset="UTF-8">
# 使用ONNX-MLIR-Serving推理MNIST模型

## 教程

这个教程演示了如何使用ONNX-MLIR-Serving框架对MNIST模型进行推理。
按照以下步骤设置环境并执行推断。

### 步骤1：构建ONNX-MLIR-Serving

1. 首先构建ONNX-MLIR-Serving Docker镜像。进入`onnx-mlir-serving`目录并运行以下命令:

```shell
cd onnx-mlir-serving
sudo docker build -t onnx/onnx-mlir-server .
```

### 步骤2：获取MNIST ONNX模型

1. 从ONNX Model Zoo下载MNIST模型及其相关输入数据集。使用以下命令下载并提取模型：

```shell
wget https://github.com/onnx/models/raw/main/vision/classification/mnist/model/mnist-12.tar.gz

tar xvzf mnist-12.tar.gz

```
这将创建`mnist-12`目录，其中包含模型及其关联的数据集。

### 第三步：使用ONNX-MLIR编译ONNX模型

1. 使用 `onnx/onnx-mlir-server` 镜像启动一个 Docker 容器。使用以下命令运行容器并打开其内部的 shell 终端：
（ "container#" 的意思是容器里面的shell）

```shell
sudo docker run -it --name serving onnx/onnx-mlir-server
```

2. 在另一个终端会话中，使用以下命令将主机上的`mnist-12`目录复制到正在运行的容器中：

```
sudo docker cp mnist-12/ serving:/workdir
```

3. 返回容器的 shell 环境并导航到 `mnist-12` 目录。使用以下命令将 `mnist-12.onnx` 模型编译为可执行代码：

```shell
container# cd mnist-12/
container# onnx-mlir -O3 --maccel=NNPA mnist-12.onnx
```
这将在`mnist-12`目录中生成一个名为`mnist-12.so`的共享库文件。

4. 验证文件是否成功生成：

```shell
container# ls
mnist-12.onnx  mnist-12.so  test_data_set_0
mnist-12.onnx mnist-12.so test_data_set_0
```

### 步骤 4：配置用于 GRPC 服务的模型
在`grpc_server`的`models`目录中创建一个模型的目录结构。

   ```
   grpc_server
   ├── models
   │   └── mnist
   │       ├── config
   │       ├── model.so
   │       └── model.onnx
   ```

1. 将模型的ONNX和编译的`.so`文件复制到容器内的`models/mnist`目录中：

```shell
container＃cd / workdir / onnx-mlir-serving / cmake / build
container# mkdir -p models/mnist
container# cd models/mnist/
container＃cp /workdir/mnist-12/mnist-12.onnx model.onnx
container# cp /workdir/mnist-12/mnist-12.so model.so
```
2.根据ONNX文件为模型生成一个配置文件:

```shell
container# ../../utils/OnnxReader ./model.onnx
container# ls
config  model.onnx  model.so
```

`config`文件应采用JSON格式，并包含有关模型的输入、输出和最大批处理大小的信息。

### 步骤5：启动GRPC服务器

1. 返回到`grpc_server`目录并启动服务器：
   ```shell
   container# cd /workdir/onnx-mlir-serving/cmake/build
   container# ./grpc_server
   wait time 0ns
   batch max size 1
   thread number 1
   Server listening on 0.0.0.0:50051
   ```

### 步骤 6：使用客户端发送推理请求

开始一个客户端发送推论请求，对MNIST模型进行推论。使用提供的C++客户端示例`Client`，执行以下命令：
 ```shell
   container# cd /workdir/onnx-mlir-serving/cmake/build/cpp
   container# ./Client /workdir/mnist-12 mnist
   ```

输出将是一个表示基于输入的每个数字（0-9）可能性的数组。推理结果将是具有最高可能性的数字。

```
结果大小：10
-48.8125
-4.6875
-21.9062
-13.3906
75.25
-8.10938
-52.3125
-52.3125
22.625
-16.6875
48.0625
```
在这个例子中，第5个数字（第4位数）的最高可能性为75.25，表明数字4是预测结果。

注意：请根据您的特定设置和环境调整路径和命令。

该程序发送mnist数据集的第一条记录（input*.pb）以获取推理结果。
如果你愿意，你可以更新示例客户端代码以读取其他数据集或你自己的记录。

### 客户端示例代码

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
