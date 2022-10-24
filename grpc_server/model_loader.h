#pragma once
#pragma GCC diagnostic ignored "-Wunused-parameter"
#ifndef ONNXMLIR_MODEL_LOADER_H
#define ONNXMLIR_MODEL_LOADER_H
#include <dlfcn.h>
#include <queue>
#include <thread>
#include <pthread.h>
#include <chrono>
#include <atomic>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <fcntl.h>
#include <unistd.h>
#include "OnnxMlirRuntime.h"
#include "utils/inference.grpc.pb.h"

using google::protobuf::io::FileOutputStream;
using google::protobuf::io::FileInputStream;

using inference::InferenceRequest;
using std::chrono::high_resolution_clock;
using Task = std::function<void(std::function<void(std::string)>)>;

extern std::chrono::high_resolution_clock::time_point originTime;

typedef struct tensorInfo_
{
  int32_t data_type;
  std::vector<int64_t> shape;
  int64_t batch_dim;
} TensorInfo;


//TODO bfloat not support for onnx-mlir.
const std::map<onnx::TensorProto_DataType, OM_DATA_TYPE> ONNX_DATA_TYPE_TO_OM = {
    {onnx::TensorProto_DataType_BOOL, ONNX_TYPE_BOOL},   // bool  -> BOOL  // char  -> INT8 (platform dependent, can be UINT8)
    {onnx::TensorProto_DataType_INT8, ONNX_TYPE_INT8},   // int8_t   -> INT8
    {onnx::TensorProto_DataType_UINT8, ONNX_TYPE_UINT8},  // uint8_t  -> UINT8,  unsigned char  -> UNIT 8
    {onnx::TensorProto_DataType_INT16, ONNX_TYPE_INT16},  // int16_t  -> INT16,  short          -> INT16
    {onnx::TensorProto_DataType_UINT16, ONNX_TYPE_UINT16}, // uint16_t -> UINT16, unsigned short -> UINT16
    {onnx::TensorProto_DataType_INT32, ONNX_TYPE_INT32},  // int32_t  -> INT32,  int            -> INT32
    {onnx::TensorProto_DataType_UINT32, ONNX_TYPE_UINT32}, // uint32_t -> UINT32, unsigned int   -> UINT32
    {onnx::TensorProto_DataType_INT64, ONNX_TYPE_INT64},  // int64_t  -> INT64,  long           -> INT64
    {onnx::TensorProto_DataType_UINT64, ONNX_TYPE_UINT64}, // uint64_t -> UINT64, unsigned long  -> UINT64
    {onnx::TensorProto_DataType_FLOAT, ONNX_TYPE_FLOAT},  // float    -> FLOAT
    {onnx::TensorProto_DataType_DOUBLE, ONNX_TYPE_DOUBLE}, // double   -> DOUBLE
    {onnx::TensorProto_DataType_STRING, ONNX_TYPE_STRING},    // const char * -> STRING
    {onnx::TensorProto_DataType_COMPLEX64, ONNX_TYPE_COMPLEX64},  // _Complex float -> COMPLEX64
    {onnx::TensorProto_DataType_COMPLEX128, ONNX_TYPE_COMPLEX128}, // _Complex double -> COMPLEX128
};

const std::map<OM_DATA_TYPE, onnx::TensorProto_DataType> OM_TO_ONNX_DATA_TYPE = {
    {ONNX_TYPE_BOOL, onnx::TensorProto_DataType_BOOL,},   // bool  -> BOOL  // char  -> INT8 (platform dependent, can be UINT8)
    {ONNX_TYPE_INT8, onnx::TensorProto_DataType_INT8},   // int8_t   -> INT8
    {ONNX_TYPE_UINT8, onnx::TensorProto_DataType_UINT8},  // uint8_t  -> UINT8,  unsigned char  -> UNIT 8
    {ONNX_TYPE_INT16, onnx::TensorProto_DataType_INT16},  // int16_t  -> INT16,  short          -> INT16
    {ONNX_TYPE_UINT16, onnx::TensorProto_DataType_UINT16}, // uint16_t -> UINT16, unsigned short -> UINT16
    {ONNX_TYPE_INT32, onnx::TensorProto_DataType_INT32},  // int32_t  -> INT32,  int            -> INT32
    {ONNX_TYPE_UINT32, onnx::TensorProto_DataType_UINT32}, // uint32_t -> UINT32, unsigned int   -> UINT32
    {ONNX_TYPE_INT64, onnx::TensorProto_DataType_INT64},  // int64_t  -> INT64,  long           -> INT64
    {ONNX_TYPE_UINT64, onnx::TensorProto_DataType_UINT64}, // uint64_t -> UINT64, unsigned long  -> UINT64
    {ONNX_TYPE_FLOAT, onnx::TensorProto_DataType_FLOAT},  // float    -> FLOAT
    {ONNX_TYPE_DOUBLE, onnx::TensorProto_DataType_DOUBLE, }, // double   -> DOUBLE
    {ONNX_TYPE_STRING, onnx::TensorProto_DataType_STRING},    // const char * -> STRING
    {ONNX_TYPE_COMPLEX64, onnx::TensorProto_DataType_COMPLEX64},  // _Complex float -> COMPLEX64
    {ONNX_TYPE_COMPLEX128, onnx::TensorProto_DataType_COMPLEX128}, // _Complex double -> COMPLEX128
};


class AbstractCallData
{
public:
  virtual InferenceRequest &getRequestData() = 0;
  // virtual std::vector<TensorData> getInputsData() = 0;
  virtual void sendBack() = 0;
  // virtual void AddInputs(TensorData data) = 0;
  virtual  onnx::TensorProto* AddOutputTensor() = 0;
  high_resolution_clock::time_point now;
};

class OnnxMlirModelLoader
{
public:
  bool LoadModel(char *model_path);


  OMTensor *(*dll_omTensorCreate)(void *, int64_t *, int64_t, OM_DATA_TYPE);
  OMTensor *RunModel(void *x1Data, int64_t *shape, int64_t rank, OM_DATA_TYPE type);
  OMTensorList *RunModel(OMTensor **list, int);

  bool success{false};

private:
  OMTensorList *(*dll_run_main_graph)(OMTensorList *);
  const char *(*dll_omInputSignature)();
  const char *(*dll_omOutputSignature)();
  OMTensorList *(*dll_omTensorListCreate)(OMTensor **, int);
  OMTensor *(*dll_omTensorListGetOmtByIndex)(OMTensorList *, int64_t);
  void *(*dll_omTensorGetDataPtr)(OMTensor *);
  void (*dll_omTensorDestroy)(OMTensor *tensor);
  void (*dll_omTensorListDestroy)(OMTensorList *);
};

typedef struct logInfo_
{
  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
  std::string key;
  int64_t inference_size;
} LogInfo;



class OnnxMlirModel
{
public:
  OnnxMlirModel(const char *_model_name);

  void ReadConfigFile(char *fileName);

  void ReadModelConfigFile(char *file_path);

  bool CheckInputData(AbstractCallData *data);

  void AddInferenceData(AbstractCallData *data);

  bool Ready(int wait, int max_batchsize_);

  std::string Calulate_duration(high_resolution_clock::time_point time1, high_resolution_clock::time_point time2)
  {
    return std::to_string(std::chrono::duration<double, std::nano>(time1 - time2).count());
  }

  void Add_log(LogInfo info, std::function<void(std::string)> log);

  Task Perpare_and_run(AbstractCallData *data);

  Task Perpare_and_run(int64_t batchsize_);

  char model_name[50];
  OnnxMlirModelLoader loader;
  std::queue<AbstractCallData *> inference_data;
  int max_batchsize = -1;
  int batch_dim = -1;
  char typeName[5];
  std::vector<TensorInfo> inputs;
  std::vector<TensorInfo> outputs;

private:
  std::mutex lock_;
};

#endif