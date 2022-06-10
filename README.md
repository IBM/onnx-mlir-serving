grpc source code: /aivol/grpc/

grpc build: /aivol/grpc_install
## Prerequisite

GPRC Installed

https://github.com/grpc/grpc/blob/master/BUILDING.md#build-from-source

GPRC Installation DIR example: grpc/cmake/install

ONNX MLIR Build is built

Pls copy include files from onnx-mlir source to onnx-mlir build dir.

```
ls aiu/onnx-mlir-build/*
aiu/onnx-mlir-build/include:
benchmark  CMakeLists.txt  google  onnx  onnx-mlir  OnnxMlirCompiler.h  OnnxMlirRuntime.h  rapidcheck  rapidcheck.h

aiu/onnx-mlir-build/lib:
libcruntime.a
```

## Build:

```
cmake -DGRPC_DIR:STRING=${GPRC_SRC_DIR} -DONNX_COMPILER_BUILD_DIR:STRING${ONNX_MLIR_BUILD_DIR} -DLOADGEN_DIR:STRING=~/code/aiu/inference/loadgen -DCMAKE_PREFIX_PATH=/aiu/grpc/cmake/install ../..
make -j
```

## run:

server:
```
./AIU_async_server <wait time ns> <num of thread pool>
```
client:
```
./AIU_async_client <file path> 
```
baching run:
```
./app <input dir> <1 for grpc call, 0 for local call> <target_qps> <useQueue> <num of thread>
```

example:
1.for accuracy run only (for UT)
```
./app /aivol/inputs/ccf1_inputs 1
```
2.for batching grpc call
```
./app /aivol/inputs/ccf1_inputs 1 1000 0 1000
```

# Docker-Build

1. Build Base
```
docker build -f Dockerfile.base -t onnx/aigrpc-base .
```
2. Build AI GPRC Server and Client
```
docker build -t onnx/aigrpc-server .
```
