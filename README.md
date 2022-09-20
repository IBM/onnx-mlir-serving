# ONNX-MLIR Serving

This project implements a GRPC server written with C++ to serve onnx-mlir compiled models. Benefiting from C++ implementation, ONNX Serving has very low latency overhead and high throughput. 

ONNX Servring provides dynamic batch aggregation and workers pool feature to fully utilize AI accelerators on the machine.


## Setup ONNX-MLIR Serving on local environment


### **Prerequisite**


#### 1. GPRC Installed

[Build GRPC from Source](https://github.com/grpc/grpc/blob/master/BUILDING.md#build-from-source)

**GPRC Installation DIR example**: grpc/cmake/install


#### 2. ONNX MLIR Build is built

Copy include files from onnx-mlir source to onnx-mlir build dir.

```
ls onnx-mlir-serving/onnx-mlir-build/*
onnx-mlir-sering/onnx-mlir-build/include:
benchmark  CMakeLists.txt  google  onnx  onnx-mlir  OnnxMlirCompiler.h  OnnxMlirRuntime.h  rapidcheck  rapidcheck.h

onnx-mlir-serving/onnx-mlir-build/lib:
libcruntime.a
```

### **Build ONNX-MLIR Serving**

```
cmake -DGRPC_DIR:STRING=${GPRC_SRC_DIR} -DONNX_COMPILER_BUILD_DIR:STRING${ONNX_MLIR_BUILD_DIR} -DCMAKE_PREFIX_PATH=grpc/cmake/install ../..
make -j
```

### **Run ONNX-MLIR Server and Client**

#### Server:
```
./grpc_server <wait time ns> <num of thread pool>
```
#### Client:
```
./grpc_client <file path> 
```
#### Baching run:
```
./grpc_client <input dir> <1 for grpc call, 0 for local call> <target_qps> <useQueue> <num of thread>
```

#### Client Examples:
1.for accuracy run only (for UT)
```
./grpc_client inputs/ccf1_inputs 1
```
2.for batching grpc call
```
./grpc_client /inputs/ccf1_inputs 1 1000 0 1000
```

## Setup ONNX-MLIR Serving on Docker environment

1. Build Base
```
docker build -f Dockerfile.base -t onnx/aigrpc-base .
```
2. Build AI GPRC Server and Client
```
docker build -t onnx/aigrpc-server .
```

## Example

See [grpc-test.cc](./tests/grpc-test.cc)

- TEST_F is a simpliest example to serve minst model.
