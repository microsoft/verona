#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>
#include <mutex>

/*
 * Example run for 4 cores: time taskset --cpu-list 0-3 ./perf-con-boc_bench2 50 5000 1000 1
 */

std::mutex *forks;
int loop_time;
int phil_count;

void philosopher_main(int phil_id, int hunger)
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
    auto end = std::chrono::system_clock::now() + usec;

    // spin
    while (std::chrono::system_clock::now() < end);

    m1->unlock();
    m2->unlock();
  }
}

int main(int argc, char **argv)
{
  if (argc != 5)
  {
    fprintf(stderr, "Usage: ./exec <phil count> <hunger> <loop time us> <should_split>\n");
    return -1;
  }

  phil_count = atoi(argv[1]);
  int hunger = atoi(argv[2]);
  loop_time = atoi(argv[3]);
  int should_split = atoi(argv[4]);

  forks = new std::mutex[(unsigned long)phil_count];

  std::vector<std::thread> philosophers;
  if (should_split)
  {
    for (int i = 0; i < phil_count; i+=2)
    {
      philosophers.push_back(std::thread([=]() { philosopher_main(i, hunger); }));
    }
    for (int i = 1; i < phil_count; i+=2)
    {
      philosophers.push_back(std::thread([=]() { philosopher_main(i, hunger); }));
    }
  }
  else
  {
    for (int i = 0; i < phil_count; i++)
    {
      philosophers.push_back(std::thread([=]() { philosopher_main(i, hunger); }));
    }
  }

  std::cout << "main thread\n";

  std::for_each(philosophers.begin(), philosophers.end(), [](std::thread &t)
      {
      t.join();
      });

  return 0;
}
