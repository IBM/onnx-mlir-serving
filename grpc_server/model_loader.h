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
#include "OnnxMlirRuntime.h"
#include "inference.grpc.pb.h"

using inference::InferenceRequest;
using std::chrono::high_resolution_clock;
using Task = std::function<void(std::function<void(std::string)>)>;

extern std::chrono::high_resolution_clock::time_point originTime;

class AbstractCallData
{
public:
  virtual InferenceRequest getRequestData() = 0;
  virtual void sendBack(float *data, int size) = 0;
  high_resolution_clock::time_point now;
};

class OnnxMlirModelLoader
{
public:
  bool LoadModel(char *model_path);

  OMTensor *RunModel(void *x1Data, int64_t *shape, int64_t rank, OM_DATA_TYPE type);

  bool success{false};

private:
  OMTensorList *(*dll_run_main_graph)(OMTensorList *);
  const char *(*dll_omInputSignature)();
  const char *(*dll_omOutputSignature)();
  OMTensor *(*dll_omTensorCreate)(void *, int64_t *, int64_t, OM_DATA_TYPE);
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

private:
  std::mutex lock_;
};

#endif