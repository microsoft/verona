#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>
#include <mutex>
#include <cassert>

/*
 * Example run for 4 cores, no-split, and automatic deadlock prevention
 * time taskset --cpu-list 0-3 ./perf-con-boc_bench2 50 1000 1000 0 0
 */

std::mutex *forks;
int loop_time;
int phil_count;

void philosopher_main_manual(int phil_id, int hunger)
{
  std::mutex *m1, *m2;
  for (int i=0;i<hunger;i++)
  {
    if (phil_id == phil_count - 1)
    {
      m1 = &forks[(phil_id+1) % phil_count];
      m2 = &forks[phil_id];
    }
    else
    {
      m1 = &forks[phil_id];
      m2 = &forks[(phil_id+1) % phil_count];
    }

    m1->lock();
    m2->lock();

    // eat -> busy wait
    std::chrono::microseconds usec(loop_time);
    auto start = std::chrono::system_clock::now();
    auto end = start + usec;

    // spin
    while (std::chrono::system_clock::now() <= end);

    m1->unlock();
    m2->unlock();
  }
}

void philosopher_main(int phil_id, int hunger)
{
  for (int i=0;i<hunger;i++)
  {
    std::lock(forks[(phil_id+1) % phil_count], forks[phil_id]);
    std::lock_guard<std::mutex> lk1(forks[(phil_id+1) % phil_count], std::adopt_lock);
    std::lock_guard<std::mutex> lk2(forks[phil_id], std::adopt_lock);

    // eat -> busy wait
    std::chrono::microseconds usec(loop_time);
    auto start = std::chrono::system_clock::now();
    auto end = start + usec;

    // spin
    while (std::chrono::system_clock::now() <= end);
  }
}

int main(int argc, char **argv)
{
  if (argc != 6)
  {
    fprintf(stderr, "Usage: ./exec <phil count> <hunger> <loop time us> <should_split> <manual>\n");
    return -1;
  }

  phil_count = atoi(argv[1]);
  int hunger = atoi(argv[2]);
  loop_time = atoi(argv[3]);
  int should_split = atoi(argv[4]);
  int manual = atoi(argv[5]);

  forks = new std::mutex[(unsigned long)phil_count];

  void(*fn)(int,int);
  if (manual)
    fn = philosopher_main_manual;
  else
    fn = philosopher_main;
  
  std::vector<std::thread> philosophers;
  if (should_split)
  {
    for (int i = 0; i < phil_count; i+=2)
    {
      philosophers.push_back(std::thread([=]() { fn(i, hunger); }));
    }
    for (int i = 1; i < phil_count; i+=2)
    {
      philosophers.push_back(std::thread([=]() { fn(i, hunger); }));
    }
  }
  else
  {
    for (int i = 0; i < phil_count; i++)
    {
      philosophers.push_back(std::thread([=]() { fn(i, hunger); }));
    }
  }

  std::for_each(philosophers.begin(), philosophers.end(), [](std::thread &t)
      {
      t.join();
      });

  return 0;
}
