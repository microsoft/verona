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
  /// Protects the state of the accounts (note, in this benchmark there is no
  /// state)
  std::mutex* accounts;

  /// Protects the state of the logger (note, in this benchmark that is
  /// tx_count)
  std::mutex logger;

  /// Something representing the loggers state.
  uint64_t tx_count = 0;
}

void thread_main(size_t id)
{
  size_t a, b;
  xoroshiro::p128r64 rand;
  rand.set_state(id + 1);

  for (size_t i = 0; i < (TRANSACTIONS / NUM_WORKERS); i++)
  {
    a = rand.next() % NUM_ACCOUNTS;
    b = a;
    while (b == a)
      b = rand.next() % NUM_ACCOUNTS;

    {
      std::scoped_lock lock(accounts[a], accounts[b]);

      busy_loop(WORK_USEC);

      // Must expand lock set to ensure the order is consistent.
      std::lock_guard<std::mutex> lg_lk(logger);

      // Superficial logging.
      tx_count++;
    }
  }
}

int pthread_main()
{
  accounts = new std::mutex[NUM_ACCOUNTS];

  std::vector<std::thread> workers;
  for (size_t i = 0; i < NUM_WORKERS; i++)
  {
    workers.push_back(std::thread([=]() { thread_main(i); }));
  }

  std::for_each(
    workers.begin(), workers.end(), [](std::thread& t) { t.join(); });

  return 0;
}
