#include "thread_pool.h"

void AIInfrenceThreadPool::AddTask(Task task)
{
  lock_guard<mutex> lock(lock_);
  tasks_.push(task);
  task_cv_.notify_one();
}

void AIInfrenceThreadPool::AddThread(int _size)
{

  int size = std::min(_size, THREADPOOL_MAX_NUM);
  while(size > 0)
  {
    pool_.emplace_back([this](int cpuindex){ 
      int cpu = (cpuindex-1) * 12;
      int total_cpu_num = sysconf(_SC_NPROCESSORS_CONF);
      cpu = cpu % total_cpu_num + cpu / total_cpu_num;
      cpu_set_t mask;
      // cpu_set_t get;
      CPU_ZERO(&mask);
      CPU_SET(*&cpu,&mask);
      char tname[20];
      sprintf(tname, "Inference thread %d", cpu);
      
      prctl(PR_SET_NAME, tname);
      if(sched_setaffinity(0,sizeof(cpu_set_t),&mask)==-1)
      {
        printf("warning: could not set CPU affinity, continuing...\n");
      }
      while (run_)
      {

        Task task;
        {
        unique_lock<mutex> lock{ lock_ };
        task_cv_.wait(lock, [this]{ return !run_ || !tasks_.empty(); });

        if (!run_ && tasks_.empty())
          return;

        task = move(tasks_.front());
        tasks_.pop();
        }

        idl_thread_num_--;
        task(to_log);
        idl_thread_num_++;
      } }, size);
    size--;
    idl_thread_num_++;
  }
}

void AIInfrenceThreadPool::PrintLogs()
{
  std::cout << "print log " << std::endl;
  std::cout << log_stream_.str() << std::endl;
  log_stream_.clear();
}