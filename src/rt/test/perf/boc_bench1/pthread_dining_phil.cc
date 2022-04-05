// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "args.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <mutex>
#include <test/harness.h>
#include <thread>
#include <vector>

namespace {
  std::mutex* forks;
}

void philosopher_main(size_t phil_id, size_t hunger)
{
  for (size_t i = 0; i < hunger; i++)
  {
    std::lock(forks[(phil_id + 1) % NUM_PHILOSOPHERS], forks[phil_id]);
    std::lock_guard<std::mutex> lk1(
      forks[(phil_id + 1) % NUM_PHILOSOPHERS], std::adopt_lock);
    std::lock_guard<std::mutex> lk2(forks[phil_id], std::adopt_lock);

    busy_loop(WORK_USEC);
  }
}

int pthread_main()
{
  forks = new std::mutex[NUM_PHILOSOPHERS];

  std::vector<std::thread> philosophers;
  if (OPTIMAL_ORDER)
  {
    for (size_t i = 0; i < NUM_PHILOSOPHERS; i += 2)
    {
      philosophers.push_back(
        std::thread([=]() { philosopher_main(i, HUNGER); }));
    }
    for (size_t i = 1; i < NUM_PHILOSOPHERS; i += 2)
    {
      philosophers.push_back(
        std::thread([=]() { philosopher_main(i, HUNGER); }));
    }
  }
  else
  {
    for (size_t i = 0; i < NUM_PHILOSOPHERS; i++)
    {
      philosophers.push_back(
        std::thread([=]() { philosopher_main(i, HUNGER); }));
    }
  }

  std::for_each(
    philosophers.begin(), philosophers.end(), [](std::thread& t) { t.join(); });

  return 0;
}
