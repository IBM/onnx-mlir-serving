#pragma once
#ifndef DLC_MODEL_LOADER_H
#define DLC_MODEL_LOADER_H
#include <dlfcn.h>
#include <queue>
#include <pthread.h>
#include <chrono>
#include <atomic>
#include <vector>
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
    virtual void sendBack(float* data, int size) = 0;
    high_resolution_clock::time_point now;
};

class DLCModelLoader
{
  public:
    bool LoadModel(char* model_path){
      void *handle = dlopen(model_path, RTLD_LAZY);
      if (!handle) {
        std::cout << "  Did not find model " << model_path << std::endl;
        return false;
      }
      success = true;
      dll_run_main_graph = (OMTensorList * (*)(OMTensorList *))
      dlsym(handle, "run_main_graph");
      assert(!dlerror() && "failed to load entry point");
      dll_omInputSignature = (const char *(*)())dlsym(handle, "omInputSignature");
      assert(!dlerror() && "failed to load omInputSignature");
      dll_omOutputSignature = (const char *(*)())dlsym(handle, "omOutputSignature");
      assert(!dlerror() && "failed to load omOutputSignature");
      dll_omTensorCreate =
          (OMTensor * (*)(void *, int64_t *, int64_t, OM_DATA_TYPE))
              dlsym(handle, "omTensorCreate");
      assert(!dlerror() && "failed to load omTensorCreate");
      dll_omTensorListCreate = (OMTensorList * (*)(OMTensor **, int))
          dlsym(handle, "omTensorListCreate");
      assert(!dlerror() && "failed to load omTensorListCreate");
      dll_omTensorListGetOmtByIndex = 
          (OMTensor * (*)(OMTensorList *, int64_t)) dlsym(handle, "omTensorListGetOmtByIndex");
      dll_omTensorGetDataPtr = (void *(*)(OMTensor *)) dlsym(handle, "omTensorGetDataPtr");

      dll_omTensorListDestroy =
          (void (*)(OMTensorList *))dlsym(handle, "omTensorListDestroy");
      assert(!dlerror() && "failed to load omTensorListDestroy");
      dll_omTensorDestroy =
          (void (*)(OMTensor *))dlsym(handle, "omTensorDestroy");
      return true;
    };

    OMTensor * RunModel(void *x1Data, int64_t * shape, int64_t rank, OM_DATA_TYPE type){
      OMTensor *x1 = dll_omTensorCreate(x1Data, shape, rank, type);
      OMTensor *list[1] = {x1};
      OMTensorList *input = dll_omTensorListCreate(list,1);
      OMTensorList *outputList = dll_run_main_graph(input);

      OMTensor *y = dll_omTensorListGetOmtByIndex(outputList,0);
      omTensorDestroy(x1);
      return y;
    }



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

typedef struct logInfo_{
  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
  std::string key;
  size_t inference_size;
}LogInfo;



class DLCModel{
  public:
    DLCModel(const char* _model_name){
      batching = 0;
      // model_name = std::string(_model_name).c_str();
      strcpy(model_name, _model_name);
      char model_path[50];
      sprintf(model_path, "./models/%s/model.so", model_name);
      // std::cout<<"create " << model_path<<std::endl;
      if(!loader.LoadModel(model_path)){
        std::cout<<"create failed"<<std::endl;
        model_name[0] = 0;
      }
      if(strcmp(model_name, "ccf1")==0){
        batching = 1;
      }
    }

    void AddInferenceData(AbstractCallData *data){
      std::unique_lock<std::mutex> lock{ lock_ };
      inference_data.push(data);
    }

    bool Ready(int wait){
      bool check = false;
      int size = 0;
      double d = 0;
      {
        std::lock_guard<std::mutex> lock(lock_);

        size = inference_data.size();
        if(size>0 && wait !=0){
          high_resolution_clock::time_point pnow = inference_data.front()->now;
          high_resolution_clock::time_point now = high_resolution_clock::now();
          d = std::chrono::duration<double, std::nano>(now -pnow).count();
          double w =(double) wait; // * (float)size;
          check = d >= w;
        } 
      }
      return check;
    }

    std::string Calulate_duration(high_resolution_clock::time_point time1, high_resolution_clock::time_point time2){
      return  std::to_string(std::chrono::duration<double, std::nano>(time2 - time1).count());
    }

    void Add_log(LogInfo info, std::function<void(std::string)> log){
      std::stringstream log_stream;
      log_stream << std::this_thread::get_id() << "," << info.key << "," << info.inference_size << ",";
      log_stream << Calulate_duration(info.end, info.start) << ",";
      log_stream << Calulate_duration(info.start, originTime) << ",";
      log_stream << Calulate_duration(info.end , originTime) << std::endl;
      log(log_stream.str());
      log_stream.clear();
    }

    Task Perpare_and_run(AbstractCallData* data){

      return [this, data](std::function<void(std::string)> log){
        auto curr = data->getRequestData();
        int64_t rank = curr.shape().size();
        int64_t* shape = curr.mutable_shape()->mutable_data();
        float *x1Data = curr.mutable_data()->mutable_data();

        OMTensor *y = loader.RunModel(x1Data, shape, rank, ONNX_TYPE_FLOAT);
        float *prediction = (float *)omTensorGetDataPtr(y);
        int64_t *output_shape = omTensorGetShape(y);
        int resultsize = 1;
        for(int i = 0; i < omTensorGetRank(y); i++){
          resultsize *= output_shape[i];
        }

        data->sendBack(prediction,resultsize);

        omTensorDestroy(y);
      };

    }


    Task Perpare_and_run(int64_t batchsize_){

      return [this, batchsize_](std::function<void(std::string)> log){
        int64_t count = 0;
        std::vector<AbstractCallData*> my_queue;
        {
          std::unique_lock<std::mutex> lock{ lock_ };
          int64_t totalsize = inference_data.size();
          while(count<batchsize_ && count<totalsize){
            my_queue.push_back(inference_data.front());
            inference_data.pop();
            count ++;
          }
        }

        size_t batchsize = count;
        if(batchsize < 1){
          return ;
        }

        std::stringstream log_stream;
        high_resolution_clock::time_point pnow;
        high_resolution_clock::time_point now;

        // pnow = my_queue[0]->now;
        // now  = high_resolution_clock::now();
        // log_stream << std::this_thread::get_id() << ",wake up," << my_queue.size() << ",";
        // log_stream << Calulate_duration(pnow, now) << ",";
        // log_stream << Calulate_duration(pnow, originTime) << ",";
        // log_stream << Calulate_duration(now , originTime) << std::endl;
        // log(log_stream.str());
        // log_stream.clear();
        Add_log({pnow, now, "wake up", my_queue.size()}, log);


        // std::cout << "batch size " << batchsize << std::endl;
        pnow = high_resolution_clock::now();
        int64_t rank = 3;
        int64_t shape[3] = {7,(int64_t)batchsize,204};
        int64_t totalsize = 7* batchsize * 204;
        float *x1Data = (float*)calloc(totalsize, sizeof(float));

        for(size_t i = 0; i < batchsize; i ++){
          auto curr = my_queue[i]->getRequestData();
          for(size_t j = 0; j < 7; j++){
            for(size_t k = 0; k< 204; k++){
              x1Data[(j*batchsize+i)*204+k] = curr.data((j*204)+k);
            }
          }
        }

        OMTensor *y = loader.RunModel(x1Data, shape, rank, ONNX_TYPE_FLOAT);
        float *prediction = (float *)omTensorGetDataPtr(y);
        int64_t *output_shape = omTensorGetShape(y);
        int resultsize = 1;
        for(int i = 0; i < omTensorGetRank(y); i++){
          resultsize *= output_shape[i];
        }

        float* singleResult = (float*)calloc(batchsize*7, sizeof(float));

        // chrono::high_resolution_clock::time_point now = chrono::high_resolution_clock::now();

        // float singleResult[batch_size_*7] = {0};
        float* start = singleResult;
        for(size_t i = 0;i < batchsize; i++ ){
          for(size_t j = 0; j < 7; j++){
            start[j] = prediction[j*batchsize+i];
          }
          my_queue[i]->sendBack(start,7);
          start += 7;
        }

        now  = high_resolution_clock::now();
        double d    = std::chrono::duration<double, std::nano>(now -pnow).count();
        log_stream << std::this_thread::get_id() << ",inference," << batchsize << "," << std::to_string(d) << ",";
        log_stream << std::to_string(std::chrono::duration<double, std::nano>(pnow -originTime).count()) << ",";
        log_stream << std::to_string(std::chrono::duration<double, std::nano>(now  -originTime).count()) << std::endl;

        free(singleResult);
        free(x1Data);
        omTensorDestroy(y);

        log(log_stream.str());
      };

    }


    char model_name[50];
    DLCModelLoader loader;
    std::queue<AbstractCallData*> inference_data;
    int batching = 0;
  private:
    std::mutex lock_;

};

#endif  