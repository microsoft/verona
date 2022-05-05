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

namespace
{
  std::mutex* forks;
}

void philosopher_main(size_t phil_id, size_t hunger)
{
  auto* p1 = forks + phil_id;
  auto* p2 = forks + ((phil_id + 1) % NUM_PHILOSOPHERS);

  if (MANUAL_LOCK_ORDER)
  {
    if (p1 > p2)
      std::swap(p1, p2);
    for (size_t i = 0; i < hunger; i++)
    {
      std::scoped_lock lock1(*p1);
      std::scoped_lock lock2(*p2);
      busy_loop(WORK_USEC);
    }
  }
  else
  {
    for (size_t i = 0; i < hunger; i++)
    {
      std::scoped_lock lock(*p1, *p2);
      busy_loop(WORK_USEC);
    }
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
