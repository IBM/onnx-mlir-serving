# ONNX-MLIR Serving

This project implements a GRPC server written with C++ to serve [onnx-mlir](https://onnx.ai/onnx-mlir/) compiled models. Benefiting from C++ implementation, ONNX Serving has very low latency overhead and high throughput. 

ONNX Servring provides dynamic batch aggregation and workers pool feature to fully utilize AI accelerators on the machine.

## [ONNX-MLIR](https://onnx.ai/onnx-mlir/)
ONNX-MLIR is compiler technology to transform a valid Open Neural Network Exchange (ONNX) graph into code that implements the graph with minimum runtime support. It implements the ONNX standard and is based on the underlying LLVM/MLIR compiler technology.

## Build

1. Build ONNX-MLIR Serving on local environment
2. Build ONNX-MLIR Serving on Docker environment (Recommended)

### Build ONNX-MLIR Serving on local environment


#### **Prerequisite**


##### 1. GPRC Installed

[Build GRPC from Source](https://github.com/grpc/grpc/blob/master/BUILDING.md#build-from-source)

**GPRC Installation DIR example**: grpc/cmake/install


##### 2. ONNX MLIR Build is built

Copy include files from onnx-mlir source to onnx-mlir build dir.

```
ls onnx-mlir-serving/onnx-mlir-build/*
onnx-mlir-sering/onnx-mlir-build/include:
benchmark  CMakeLists.txt  google  onnx  onnx-mlir  OnnxMlirCompiler.h  OnnxMlirRuntime.h  rapidcheck  rapidcheck.h

onnx-mlir-serving/onnx-mlir-build/lib:
libcruntime.a
```

#### **Build ONNX-MLIR Serving**

```
cmake -DCMAKE_BUILD_TYPE=Release -DGRPC_DIR:STRING={GPRC_SRC_DIR} -DONNX_COMPILER_DIR:STRING={ONNX_MLIR_BUILD_DIR} -DCMAKE_PREFIX_PATH={GPRC_INSTALL_DIR} ../..
make -j
```

### Build ONNX-MLIR Serving on Docker environment


Build AI GPRC Server and Client
```
docker build -t onnx/aigrpc-server .
```


## **Run ONNX-MLIR Server and Client**

### Server:
```
./grpc_server -h
usage: grpc_server [options]
    -w arg     wait time for batch size, default is 0
    -b arg     server side batch size, default is 1
    -n arg     thread numberm default is 1

./grpc_server
```
### Add more models

Build Models Directory
```
/cmake/build
mkdir models
```
example models directory
```
models
└── mnist
    ├── grpc_config.txt
    ├── img0.data
    ├── model.so
    └── val_map.txt
```

#### grpc_config.txt
describe model configuration
```
<model name>
<input type> <sizeof input type>
<model rank>
<model shape>
<batching size or empty for no baching model>
<batching dimension index or emoty for no baching model>
```
**\<input type\>** comes from the ONNX-MLIR
```
const std::map<std::string, OM_DATA_TYPE> OM_DATA_TYPE_CPP_TO_ONNX = {
    {"b", ONNX_TYPE_BOOL},   // bool  -> BOOL
    {"c", ONNX_TYPE_INT8},   // char  -> INT8 (platform dependent, can be UINT8)
    {"a", ONNX_TYPE_INT8},   // int8_t   -> INT8
    {"h", ONNX_TYPE_UINT8},  // uint8_t  -> UINT8,  unsigned char  -> UNIT 8
    {"s", ONNX_TYPE_INT16},  // int16_t  -> INT16,  short          -> INT16
    {"t", ONNX_TYPE_UINT16}, // uint16_t -> UINT16, unsigned short -> UINT16
    {"i", ONNX_TYPE_INT32},  // int32_t  -> INT32,  int            -> INT32
    {"j", ONNX_TYPE_UINT32}, // uint32_t -> UINT32, unsigned int   -> UINT32
    {"l", ONNX_TYPE_INT64},  // int64_t  -> INT64,  long           -> INT64
    {"m", ONNX_TYPE_UINT64}, // uint64_t -> UINT64, unsigned long  -> UINT64
    {"f", ONNX_TYPE_FLOAT},  // float    -> FLOAT
    {"d", ONNX_TYPE_DOUBLE}, // double   -> DOUBLE
    {"PKc", ONNX_TYPE_STRING},    // const char * -> STRING
    {"Cf", ONNX_TYPE_COMPLEX64},  // _Complex float -> COMPLEX64
    {"Cd", ONNX_TYPE_COMPLEX128}, // _Complex double -> COMPLEX128
};
```
example ccf model
```
ccf
f 4
3
7 1 204
16
1
```

#### img0.data
binary inference input data. Should already pre-processed. 

#### val_map.txt
list input files with expected result
example
```
img0.data 1
```

#### Use Batching
There are two place to input batch size
1. In model config.txt file 
2. When start grpc_server -b [batch size]

situation_1: grpc_server without -b, defaule batch size is 1, means no batching 
situation_2: grpc_server -b <batch_size>, batch_size > 1, and model A config.txt also has batch_size, when query model A, will use the mininum batch size.
situation_3: grpc_server -b <batch_size>, batch_size > 1, and model B config.txt did not has batch_size, when query model B, will not using batching.


### Client:
```
./Client <inputs dir> 
```
```
inputs dir
    ├── grpc_config.txt
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

The example mnist model input data in generated with little-endian, if server is in s390, need to change to big-endian before sending the gRPC request.
In the utils/grpc_client.h has the code do the change thing, remove it if your server is in x86.


## Example

See [grpc-test.cc](./tests/grpc-test.cc)

- TEST_F is a simpliest example to serve minst model.

