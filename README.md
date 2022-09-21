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
./grpc_server -h
usage: grpc_server [options]
    -w arg     wait time for batch, default is 0
    -b arg     server side batch size, default is 1
    -n arg     number of inference threads, default is 1. On z16, using 2 * chip number for dedicated lpar configuration got best result. 

./grpc_server
```
### Add more models
```
/cmake/build
mkdir models
```
example models directory
```
models
└── mnist
    ├── config.txt
    ├── img0.data
    ├── model.so
    └── val_map.txt
```

#### config.txt
discripte model configs
```
<model name>
<model rank>
<model shape>
<batching size or empty for no baching model>
<batching dimension index or empty for no baching model>
```
example ccf model
```
ccf
3
7 1 204
16
1
```
The following two files are for performance benchmark tests only.

#### img0.data

binary inference input data. Should already be pre-processed. 

#### val_map.txt

list input files with expected result
example
```
img0.data 1
```

### Enable Batching

There are two places to input batch size
1. In model config.txt file 
2. When start grpc_server -b [batch size]

situation_1: grpc_server without -b, defaule batch size is 1, means no batching 
situation_2: grpc_server -b <batch_size>, batch_size > 1, and model A config.txt also has batch_size, server will use the mininum batch size.
situation_3: grpc_server -b <batch_size>, batch_size > 1, and model B config.txt did not has batch_size, server will not use batching.


### Client:
```
./Client <inputs dir> 
```
```
inputs dir
    ├── config.txt
    ├── img0.data
    └── val_map.txt
```
#### Benchmark:
```
./Benchmark <input dir> <logfile prefix> <num of thread, for test batching>
```

#### Client Examples:
1.for accuracy run only (for UT)
```
./Client inputs/ccf1_inputs
```
2.for batching benchmark call
```
./Benchmark /inputs/ccf1_inputs ccf 16
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

