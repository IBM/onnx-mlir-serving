#pragma once
#ifndef AI_INFERENCE_THREAD_POOL_H
#define AI_INFERENCE_THREAD_POOL_H

#include <vector>
#include <queue>
#include <atomic>
#include <future>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <thread>
#include <pthread.h>
#include <stdexcept>
#include <sys/prctl.h>

#include <iostream>
#include <sstream>
#include <memory>
#include <string>


#include "model_loader.h"



using std::atomic;
using std::condition_variable;
using std::lock_guard;
using std::mutex;
using std::queue;
using std::string;
using std::thread;
using std::unique_lock;
using std::vector;
// using std::chrono::high_resolution_clock;

#define THREADPOOL_MAX_NUM 20

extern std::chrono::high_resolution_clock::time_point originTime;
extern OnnxMlirModelLoader modelLoder;


class AIInfrenceThreadPool
{
  using Task = std::function<void(std::function<void(std::string)>)>;
  vector<thread> pool_;
  queue<Task> tasks_;
  mutex lock_;
  mutex log_mutex_;
  condition_variable task_cv_;
  atomic<bool> run_{true};
  atomic<int> idl_thread_num_{0};
  int wait_ = 0;
  std::stringstream log_stream_;
  int batch_size_ = 1;

public:
  AIInfrenceThreadPool(unsigned short size)
  {
    AddThread(size);
  }
  ~AIInfrenceThreadPool()
  {
    run_ = false;
    task_cv_.notify_all();
    for (thread &thread : pool_)
    {
      if (thread.joinable())
        thread.join();
    }
  }

  void AddThread(unsigned short size);
  void AddCallData(AbstractCallData *data);
  void AddTask(Task task);
  void PrintLogs();
  int IdlCount() { return idl_thread_num_; }
  int ThreadCount() { return pool_.size(); }
  std::function<void(std::string)> to_log = [this](std::string c)
  {
    lock_guard<mutex> lock(log_mutex_);
    log_stream_ << c;
  };
};

#endif