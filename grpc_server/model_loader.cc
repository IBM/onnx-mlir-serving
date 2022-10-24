#include "model_loader.h"

inline void buildTensorProto(void *prediction,int bufferSize, OM_DATA_TYPE type, onnx::TensorProto* tensor_proto){

  int64_t typeSize = getDataTypeSize(type);
  onnx::TensorProto_DataType tensor_pType = OM_TO_ONNX_DATA_TYPE.at(type);
  tensor_proto->set_data_type(tensor_pType);
  switch(tensor_pType){
    case(onnx::TensorProto_DataType_FLOAT):
    case(onnx::TensorProto_DataType_COMPLEX64):
      tensor_proto->mutable_float_data()->Add((float*)prediction, (float*)prediction + bufferSize/typeSize);
      break;
    case(onnx::TensorProto_DataType_UINT8):
    case(onnx::TensorProto_DataType_INT8):
    case(onnx::TensorProto_DataType_UINT16):
    case(onnx::TensorProto_DataType_INT16):
    case(onnx::TensorProto_DataType_FLOAT16):
    case(onnx::TensorProto_DataType_INT32):
    case(onnx::TensorProto_DataType_BOOL):
      tensor_proto->mutable_int32_data()->Add((int32_t*)prediction, (int32_t*)prediction + bufferSize/typeSize);
      break;
    case(onnx::TensorProto_DataType_INT64):
      tensor_proto->mutable_int64_data()->Add((int64_t*)prediction, (int64_t*)prediction + bufferSize/typeSize);
      break;
    case(onnx::TensorProto_DataType_STRING):
      tensor_proto->mutable_string_data()->Add((char*)prediction, (char*)prediction + bufferSize/typeSize);
      break;
    case(onnx::TensorProto_DataType_DOUBLE):
    case(onnx::TensorProto_DataType_COMPLEX128):
      tensor_proto->mutable_double_data()->Add((double*)prediction, (double*)prediction + bufferSize/typeSize);
      break;
    case(onnx::TensorProto_DataType_UINT32):
    case(onnx::TensorProto_DataType_UINT64):
      tensor_proto->mutable_uint64_data()->Add((uint64_t*)prediction, (uint64_t*)prediction + bufferSize/typeSize);
      break;
    case(onnx::TensorProto_DataType_BFLOAT16):
      tensor_proto->set_raw_data(prediction, bufferSize);
      break;
  }

  // float* re = (float*)prediction;
  // for(int i = 0 ; i<bufferSize/typeSize; i++ ){
  //   std::cout << re[i] <<std::endl;
  // }

  return;
}


inline int64_t getTensorProtoData(const onnx::TensorProto& tensor){
  int64_t data_size;

  switch(tensor.data_type()){
    case(onnx::TensorProto_DataType_FLOAT):
    case(onnx::TensorProto_DataType_COMPLEX64):
      data_size = tensor.float_data_size();
      break;
    case(onnx::TensorProto_DataType_UINT8):
    case(onnx::TensorProto_DataType_INT8):
    case(onnx::TensorProto_DataType_UINT16):
    case(onnx::TensorProto_DataType_INT16):
    case(onnx::TensorProto_DataType_FLOAT16):
    case(onnx::TensorProto_DataType_INT32):
    case(onnx::TensorProto_DataType_BOOL):
      data_size = tensor.int32_data_size();
      break;
    case(onnx::TensorProto_DataType_INT64):
      data_size = tensor.int64_data_size();
      break;
    case(onnx::TensorProto_DataType_STRING):
      data_size = tensor.string_data_size();
      break;
    case(onnx::TensorProto_DataType_DOUBLE):
    case(onnx::TensorProto_DataType_COMPLEX128):
      data_size = tensor.double_data_size();
      break;
    case(onnx::TensorProto_DataType_UINT32):
    case(onnx::TensorProto_DataType_UINT64):
      data_size = tensor.uint64_data_size();
      break;
    case(onnx::TensorProto_DataType_BFLOAT16):
      data_size = tensor.raw_data().size()/sizeof(2);
      break;
  }

  return data_size;
}


inline void copyTensorData(const onnx::TensorProto &tensor, void* dst, int64_t index, int64_t length){

  void* src;
  switch(tensor.data_type()){
    case(onnx::TensorProto_DataType_FLOAT):
    case(onnx::TensorProto_DataType_COMPLEX64):
      src = (void*)(&tensor.float_data().data()[index]);
      break;
    case(onnx::TensorProto_DataType_UINT8):
    case(onnx::TensorProto_DataType_INT8):
    case(onnx::TensorProto_DataType_UINT16):
    case(onnx::TensorProto_DataType_INT16):
    case(onnx::TensorProto_DataType_FLOAT16):
    case(onnx::TensorProto_DataType_INT32):
    case(onnx::TensorProto_DataType_BOOL):
      src = (void*)(&tensor.int32_data().data()[index]);
      break;
    case(onnx::TensorProto_DataType_INT64):
      src = (void*)(&tensor.int64_data().data()[index]); 
      break;
    case(onnx::TensorProto_DataType_STRING):
      src =  (void*)(&tensor.string_data().data()[index]); 
      break;
    case(onnx::TensorProto_DataType_DOUBLE):
    case(onnx::TensorProto_DataType_COMPLEX128):
      src =  (void*)(&tensor.double_data().data()[index]);
      break;
    case(onnx::TensorProto_DataType_UINT32):
    case(onnx::TensorProto_DataType_UINT64):
      src =  (void*)(&tensor.uint64_data().data()[index]); 
      break;
    case(onnx::TensorProto_DataType_BFLOAT16):

      src = (void*)(&tensor.raw_data().c_str()[index]);
      break;
  }

  memcpy(dst, src, length);
}


bool OnnxMlirModelLoader::LoadModel(char *model_path)
{
  void *handle = dlopen(model_path, RTLD_LAZY);
  if (!handle)
  {
    std::cout << "Did not find model " << model_path << std::endl;
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

OMTensorList *OnnxMlirModelLoader::RunModel(OMTensor **list, int count)
{
  OMTensorList *input = dll_omTensorListCreate(list, count);
  OMTensorList *outputList = dll_run_main_graph(input);
  return outputList;
}

OnnxMlirModel::OnnxMlirModel(const char *_model_name)
{
  max_batchsize = -1;

  strcpy(model_name, _model_name);
  char model_path[70];
  sprintf(model_path, "./models/%s/model.so", model_name);

  if (!loader.LoadModel(model_path))
  {
    std::cout << "create failed" << std::endl;
    model_name[0] = 0;
  }
  char model_config[70];


  char model_onnx[70];
  sprintf(model_onnx, "./models/%s/config", model_name);
  ReadModelConfigFile(model_onnx);
}

void to_vector(const ::google::protobuf::RepeatedPtrField< ::onnx::ValueInfoProto > &info, std::vector<TensorInfo> *infos){
  // std::vector<TensorInfo> infos;
  for (auto input_data: info)
  {
    TensorInfo input_info;
    input_info.data_type = input_data.type().tensor_type().elem_type();
    auto shape = input_data.type().tensor_type().shape();
    input_info.batch_dim = -1;

    if (shape.dim_size() != 0)
    {
      int size = shape.dim_size();
      for (int i = 0; i < size; ++i)
      {
        auto dim = shape.dim(i);
        switch (dim.value_case())
        {
        case onnx::TensorShapeProto_Dimension::ValueCase::kDimParam:
          input_info.shape.emplace_back(-1);
          input_info.batch_dim = i;
          break;
        case onnx::TensorShapeProto_Dimension::ValueCase::kDimValue:
          input_info.shape.emplace_back(dim.dim_value());
          break;
        default:
          assert(false && "should never happen");
        }        
      }
    }
    infos->emplace_back(input_info);
  }
}

void OnnxMlirModel::ReadModelConfigFile(char *file_path){
  inference::ModelConfig modelConfig;

  int fd = open(file_path, O_RDONLY);
  FileInputStream* input_stream = new FileInputStream(fd);

  google::protobuf::TextFormat::Parse(input_stream, &modelConfig);
  input_stream->Close();
  close(fd);

  to_vector(modelConfig.input(), &inputs);
  to_vector(modelConfig.output(), &outputs);
  max_batchsize = modelConfig.max_batch_size();
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
  int typeSize;
  fp2 >> modelName;
  fp2 >> typeName >> typeSize;
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

bool OnnxMlirModel::CheckInputData(AbstractCallData *data){
  bool match = false;
  do{
    int input_size = data->getRequestData().tensor_size();
    if (inputs.size() != input_size)
      break;
    
    size_t count = 0;
    for(size_t count= 0; count < input_size; count ++){

      const onnx::TensorProto& tensor = data->getRequestData().tensor(count);
      size_t dim_size = tensor.dims_size();
      if(dim_size != inputs[count].shape.size())
        break;

      if(tensor.data_type() != inputs[count].data_type){
        break;
      }

      size_t data_length = 1;
      for(size_t i=0; i<dim_size; i++){
        if (inputs[count].shape[i] != -1 && inputs[count].shape[i] != tensor.dims(i)){
          break;  
        }
        data_length *= tensor.dims(i);
      }


      int64_t data_size = getTensorProtoData(tensor);

      if(data_size!=data_length)
        break;

    }
    match = true;
  }while(false);

  return match;

}

void OnnxMlirModel::AddInferenceData(AbstractCallData *data)
{
  if (!CheckInputData(data)){
    data->sendBack();
    return ;
  }
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

Task OnnxMlirModel::Perpare_and_run(AbstractCallData *callData)
{

  if (!CheckInputData(callData)){
    return [this, callData](std::function<void(std::string)> log)
    {
      callData->sendBack();
    };
  }

  return [this, callData](std::function<void(std::string)> log)
  {
    std::stringstream log_stream;
    high_resolution_clock::time_point pnow = callData->now;
    high_resolution_clock::time_point now = high_resolution_clock::now();

    Add_log({pnow, now, "wake up", 1}, log);

    int input_size = callData->getRequestData().tensor_size();
    OMTensor* tensorlist[input_size];
    for(size_t index=0; index < input_size; index++){      
      const onnx::TensorProto& tensor = callData->getRequestData().tensor(index);
      int64_t rank = tensor.dims_size(); 
      const int64_t *shape = tensor.dims().data(); 
      OM_DATA_TYPE type = ONNX_DATA_TYPE_TO_OM.at(onnx::TensorProto_DataType(tensor.data_type()));
      OMTensor *omTensor = omTensorCreateEmpty(const_cast<int64_t*>(shape), rank, type);
      void *data = omTensorGetDataPtr(omTensor);
      int64_t buffsize = omTensorGetBufferSize(omTensor);

      copyTensorData(tensor, data, 0, buffsize);
      tensorlist[index] = omTensor;
    }

    OMTensorList *yList = loader.RunModel(tensorlist,input_size);
    int result_size = omTensorListGetSize(yList);

    for(size_t index = 0; index< result_size; index++){

      OMTensor* y = omTensorListGetOmtByIndex(yList, index);
      int buffsize = omTensorGetBufferSize(y);
      int rank = omTensorGetRank(y);
      int64_t *shape = omTensorGetShape(y);
      void *prediction = (void*)omTensorGetDataPtr(y);
      OM_DATA_TYPE type = omTensorGetDataType(y);

      onnx::TensorProto* tensor = callData->AddOutputTensor();
      buildTensorProto(prediction, buffsize, type, tensor);
      tensor->mutable_dims()->Add(shape, shape+rank);
      omTensorDestroy(y);
    }

    callData->sendBack();

    for(auto x: tensorlist){
      omTensorDestroy(x);
    }

    high_resolution_clock::time_point now1 = high_resolution_clock::now();
    Add_log({now, now1, "inference", 1}, log);
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
      {
        int totalsize = inference_data.size();
        while (count < maxBatchsize && count < totalsize)
        {
          my_queue.push_back(inference_data.front());
          inference_data.pop();
          count++;
        }
      }
    }


    int64_t batchsize = count;
    if (batchsize < 1)
    {
      return;
    }

    std::stringstream log_stream;
    high_resolution_clock::time_point pnow = my_queue[0]->now;
    high_resolution_clock::time_point now = high_resolution_clock::now();

    Add_log({pnow, now, "wake up", batchsize}, log);

    pnow = high_resolution_clock::now();

    // merage
    int input_size =  my_queue[0]->getRequestData().tensor_size();

    OMTensor* tensorlist[input_size];
    for(size_t index = 0; index< input_size; index++){
      TensorInfo& info = inputs[index];
      int64_t batch_dim = 0;
      if(info.batch_dim >= 0){
        batch_dim = info.batch_dim;
      }

      const onnx::TensorProto& tensor = my_queue[0]->getRequestData().tensor(index);
      int64_t rank = tensor.dims_size(); //input.rank;
      const int64_t *single_shape = tensor.dims().data(); //input.shape;
      OM_DATA_TYPE type = ONNX_DATA_TYPE_TO_OM.at(onnx::TensorProto_DataType(tensor.data_type()));

      int64_t typeSize = getDataTypeSize(type);

      int64_t shape[rank];

      // int totalsize = 1;
      for (int64_t i = 0; i < rank; i++)
      {
        shape[i] = my_queue[0]->getRequestData().tensor(index).dims(i); //single_shape[i];
      }

      shape[batch_dim] = batchsize;

      OMTensor *omTensor = omTensorCreateEmpty(shape, rank, type);
      void *xData = omTensorGetDataPtr(omTensor);


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

      size_t b = 0;
      for(AbstractCallData* callData: my_queue){

        const onnx::TensorProto& tensor = callData->getRequestData().tensor(index);


        for (int j = 0; j < before; j++)
        {
          void * dst = xData + (j * batchsize + b) * after*typeSize;   
          copyTensorData(tensor, dst, j * after, after*typeSize);
        }
        b++;

      }
      tensorlist[index] = omTensor;
    }
    

    now = high_resolution_clock::now();
    Add_log({pnow, now, "merge", batchsize}, log);
    OMTensorList *yList = loader.RunModel(tensorlist,input_size);
    int result_size = omTensorListGetSize(yList);
    for(size_t index = 0; index< result_size; index++){
      
      // get batch dim
      TensorInfo info = outputs[index];
      int64_t batch_dim = 0;
      if(info.batch_dim >= 0){
        batch_dim = info.batch_dim;
      }

      OMTensor* y = omTensorListGetOmtByIndex(yList, index);
      int buffsize = omTensorGetBufferSize(y);
      int rank = omTensorGetRank(y);
      int64_t *shape = omTensorGetShape(y);
      uint8_t *prediction = (uint8_t*)omTensorGetDataPtr(y);
      OM_DATA_TYPE type = omTensorGetDataType(y);
      int64_t typeSize = getDataTypeSize(type);
      int singleBufferSize = buffsize / batchsize;


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
      shape[batch_dim] =1;

      size_t b = 0;
      for(AbstractCallData* callData: my_queue){
        onnx::TensorProto* tensor = callData->AddOutputTensor();
        uint8_t single_result[singleBufferSize];
        for (int j = 0; j < before; j++)
        {
          uint8_t * src = &prediction[(j * batchsize + b) * after*typeSize];
          uint8_t * dst = &single_result[j * after*typeSize];
          memcpy(dst, src, after*typeSize);
        }
        b++;

        buildTensorProto(single_result, singleBufferSize, type, tensor);
        tensor->mutable_dims()->Add(shape, shape+rank);
      }
      omTensorDestroy(y);
    }

    for(AbstractCallData* callData: my_queue){
      callData->sendBack();
    }

    now = high_resolution_clock::now();
    Add_log({pnow, now, "inference", batchsize}, log);
    for(auto x: tensorlist){
      omTensorDestroy(x);
    }

    log(log_stream.str());
  };
}
