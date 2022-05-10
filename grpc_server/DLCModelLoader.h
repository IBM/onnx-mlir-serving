#pragma once
#ifndef DLC_MODEL_LOADER_H
#define DLC_MODEL_LOADER_H
#include <dlfcn.h>
#include "OnnxMlirRuntime.h"

class DLCModelLoader
{
  public:
    void LoadModel(char* model_path){
      void *handle = dlopen(model_path, RTLD_LAZY);
      if (!handle) {
        std::cout << "  Did not find model " << model_path << std::endl;
        return ;
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

#endif  