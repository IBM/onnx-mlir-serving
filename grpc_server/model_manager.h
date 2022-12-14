#include "model_loader.h"
#include "thread_pool.h"

class OnnxMlirModelManager
{
public:
  OnnxMlirModelManager(int batch_size, int thread_num, int wait_time) : 
    tpool_(thread_num), 
    checkBatchingThread_([this]{ checkBatching(); })
  {
    batch_size_ = batch_size;
    wait_time_ = wait_time;
  }

  ~OnnxMlirModelManager()
  {
    run_ = 0;
    if (checkBatchingThread_.joinable())
    {
      checkBatchingThread_.join();
    }
  }

  int AddModel(AbstractCallData *data)
  {
    const char *model_name = data->getRequestData().model_name().c_str();
    OnnxMlirModel *model = NULL;
    {
      lock_guard<mutex> lock(lock_);
      model = Get_model(model_name);
    }

    if (model == NULL)
    {
      data->sendBack();
      return 0;
    }

    // no batching, add task to thread pool right now
    if (model->max_batchsize <= 1 || batch_size_ == 1)
    {
      tpool_.AddTask(model->Perpare_and_run(data));
    }
    // else add data to inference queue, wait batching
    else
    {
      model->AddInferenceData(data);
    }

    return 1;
  }

  void PrintLogs()
  {
    tpool_.PrintLogs();
  }

private:
  void checkBatching()
  {
    while (run_)
    {
      {
        lock_guard<mutex> lock(lock_);
        for (size_t i = 0; i < models_.size(); i++)
        {
          OnnxMlirModel *model = models_.at(i);
          if (model->max_batchsize > 0 && model->Ready(wait_time_, batch_size_))
          {
            tpool_.AddTask(model->Perpare_and_run(batch_size_));
          }
        }
      }
      std::this_thread::sleep_for(std::chrono::nanoseconds((int)(10000)));
    }
  }

  OnnxMlirModel *Get_model(const char *model_name)
  {
    OnnxMlirModel *model = NULL;
    // get model from exist model queue
    for (size_t i = 0; i < models_.size(); i++)
    {
      if (strcmp(model_name, models_[i]->model_name) == 0)
      {
        model = models_[i];
        return model;
      }
    }

    // create new model when not find
    model = new OnnxMlirModel(model_name);
    if (model->model_name[0] == 0)
    {
      return NULL;
    }
    models_.emplace_back(model);

    return model;
  }

  std::vector<OnnxMlirModel *> models_;
  AIInfrenceThreadPool tpool_;
  std::thread checkBatchingThread_;
  std::mutex lock_;
  int run_ = 1;
  int batch_size_;
  int wait_time_;
};