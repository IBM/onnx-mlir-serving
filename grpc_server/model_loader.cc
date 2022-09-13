#include "model_loader.h"

bool OnnxMlirModelLoader::LoadModel(char *model_path)
{
  void *handle = dlopen(model_path, RTLD_LAZY);
  if (!handle)
  {
    std::cout << "  Did not find model " << model_path << std::endl;
    return false;
  }
  success = true;
  dll_run_main_graph = (OMTensorList * (*)(OMTensorList *))
      dlsym(handle, "run_main_graph");
  assert(!dlerror() && "failed to load entry point");
  dll_omTensorCreate =
      (OMTensor * (*)(void *, int64_t *, int64_t, OM_DATA_TYPE))
          dlsym(handle, "omTensorCreate");
  assert(!dlerror() && "failed to load omTensorCreate");
  dll_omTensorListCreate = (OMTensorList * (*)(OMTensor **, int))
      dlsym(handle, "omTensorListCreate");
  assert(!dlerror() && "failed to load omTensorListCreate");
  dll_omTensorListGetOmtByIndex =
      (OMTensor * (*)(OMTensorList *, int64_t)) dlsym(handle, "omTensorListGetOmtByIndex");
  dll_omTensorGetDataPtr = (void *(*)(OMTensor *))dlsym(handle, "omTensorGetDataPtr");

  dll_omTensorListDestroy =
      (void (*)(OMTensorList *))dlsym(handle, "omTensorListDestroy");
  assert(!dlerror() && "failed to load omTensorListDestroy");
  dll_omTensorDestroy =
      (void (*)(OMTensor *))dlsym(handle, "omTensorDestroy");
  return true;
}

OMTensor *OnnxMlirModelLoader::RunModel(void *x1Data, int64_t *shape, int64_t rank, OM_DATA_TYPE type)
{
  OMTensor *x1 = dll_omTensorCreate(x1Data, shape, rank, type);
  OMTensor *list[1] = {x1};
  OMTensorList *input = dll_omTensorListCreate(list, 1);
  OMTensorList *outputList = dll_run_main_graph(input);

  OMTensor *y = dll_omTensorListGetOmtByIndex(outputList, 0);
  omTensorDestroy(x1);
  return y;
}

OnnxMlirModel::OnnxMlirModel(const char *_model_name)
{
  max_batchsize = -1;

  strcpy(model_name, _model_name);
  char model_path[50];
  sprintf(model_path, "./models/%s/model.so", model_name);

  if (!loader.LoadModel(model_path))
  {
    std::cout << "create failed" << std::endl;
    model_name[0] = 0;
  }
  char model_config[50];
  sprintf(model_config, "./models/%s/config.txt", model_name);
  ReadConfigFile(model_config);

}

void OnnxMlirModel::ReadConfigFile(char *fileName)
{
  std::ifstream fp2(fileName);
  if (!fp2.is_open())
  {
    printf("read model config error\n");
    return;
  }
  char modelName[20];
  int rank;
  fp2 >> modelName;
  fp2 >> rank;
  int64_t *shape = (int64_t *)malloc(rank * sizeof(int64_t));
  int index = 0;
  while (index < rank)
  {
    fp2 >> shape[index];
    index++;
  }
  fp2 >> max_batchsize;
  fp2 >> batch_dim;
  fp2.close();
}

void OnnxMlirModel::AddInferenceData(AbstractCallData *data)
{
  std::unique_lock<std::mutex> lock{lock_};
  inference_data.push(data);
}

bool OnnxMlirModel::Ready(int wait, int max_batchsize_)
{
  bool check = false;
  int size = 0;
  double d = 0;


  {
    std::lock_guard<std::mutex> lock(lock_);

    size = inference_data.size();
    if((max_batchsize_ > 1 && size >= max_batchsize_) || (size >=max_batchsize))
    {
      check = true;
    }
    else if (size > 0 && wait != 0)
    {
      high_resolution_clock::time_point pnow = inference_data.front()->now;
      high_resolution_clock::time_point now = high_resolution_clock::now();
      d = std::chrono::duration<double, std::nano>(now - pnow).count();
      double w = (double)wait; 
      check = d >= w;

    }
    else
    {
      check = true;
    }
  }
  return check;
}

void OnnxMlirModel::Add_log(LogInfo info, std::function<void(std::string)> log)
{
  std::stringstream log_stream;
  log_stream << std::this_thread::get_id() << "," << info.key << "," << info.inference_size << ",";
  log_stream << Calulate_duration(info.end, info.start) << ",";
  log_stream << Calulate_duration(info.start, originTime) << ",";
  log_stream << Calulate_duration(info.end, originTime) << std::endl;
  log(log_stream.str());
  log_stream.clear();
}

Task OnnxMlirModel::Perpare_and_run(AbstractCallData *data)
{

  return [this, data](std::function<void(std::string)> log)
  {
    auto curr = data->getRequestData();
    int64_t rank = curr.shape().size();
    int64_t *shape = curr.mutable_shape()->mutable_data();
    float *x1Data = curr.mutable_data()->mutable_data();

    OMTensor *y = loader.RunModel(x1Data, shape, rank, ONNX_TYPE_FLOAT);
    float *prediction = (float *)omTensorGetDataPtr(y);
    int64_t *output_shape = omTensorGetShape(y);
    int resultsize = 1;
    for (int i = 0; i < omTensorGetRank(y); i++)
    {
      resultsize *= output_shape[i];
    }

    data->sendBack(prediction, resultsize);

    omTensorDestroy(y);
  };
}


Task OnnxMlirModel::Perpare_and_run(int64_t maxBatchsize)
{

  return [this, maxBatchsize](std::function<void(std::string)> log)
  {
    int count = 0;
    std::vector<AbstractCallData *> my_queue;
    {
      std::unique_lock<std::mutex> lock{lock_};
      int totalsize = inference_data.size();
      while (count < maxBatchsize && count < totalsize)
      {
        my_queue.push_back(inference_data.front());
        inference_data.pop();
        count++;
      }
    }

    int batchsize = count;
    if (batchsize < 1)
    {
      return;
    }

    std::stringstream log_stream;
    high_resolution_clock::time_point pnow = my_queue[0]->now;
    high_resolution_clock::time_point now = high_resolution_clock::now();

    Add_log({pnow, now, "wake up", my_queue.size()}, log);

    pnow = high_resolution_clock::now();

    auto curr = my_queue[0]->getRequestData();
    int64_t rank = curr.shape().size();
    int64_t *shape_ = curr.mutable_shape()->mutable_data();
    int64_t *shape = (int64_t *)calloc(rank, sizeof(int64_t));

    int totalsize = 1;
    for (int64_t i = 0; i < rank; i++)
    {
      totalsize *= shape_[i];
      shape[i] = shape_[i];
    }
    totalsize *= batchsize;
    shape[batch_dim] = batchsize;
    float *x1Data = (float *)calloc(totalsize, sizeof(float));

    int before = 1;
    for (int64_t i = 0; i < batch_dim; i++)
    {
      before *= shape[i];
    }
    int after = 1;
    for (int64_t i = batch_dim + 1; i < rank; i++)
    {
      after *= shape[i];
    }

    for (size_t i = 0; i < batchsize; i++)
    {
      auto curr = my_queue[i]->getRequestData();
      // batch = i
      for (size_t j = 0; j < before; j++)
      {
        // offset = *d2
        for (size_t k = 0; k < after; k++)
        {
          // offset = (offset + b2_i ) *d3
          x1Data[(j * batchsize + i) * after + k] = curr.data((j * after) + k);
        }
      }
    }

    now = high_resolution_clock::now();
    Add_log({pnow, now, "merge", batchsize}, log);

    OMTensor *y = loader.RunModel(x1Data, shape, rank, ONNX_TYPE_FLOAT);
    float *prediction = (float *)omTensorGetDataPtr(y);
    int64_t *output_shape = omTensorGetShape(y);
    int resultsize = 1;
    for (int i = 0; i < omTensorGetRank(y); i++)
    {
      resultsize *= output_shape[i];
    }

    int singleSize = resultsize / batchsize;
    float *singleResult = (float *)calloc(batchsize * singleSize, sizeof(float));

    float *start = singleResult;
    for (int i = 0; i < batchsize; i++)
    {
      for (int j = 0; j < singleSize; j++)
      {
        start[j] = prediction[j * batchsize + i];
      }
      my_queue[i]->sendBack(start, singleSize);
      start += singleSize;
    }

    now = high_resolution_clock::now();
    Add_log({pnow, now, "inference", batchsize}, log);

    free(singleResult);
    free(x1Data);
    free(shape);
    omTensorDestroy(y);

    log(log_stream.str());
  };
}
