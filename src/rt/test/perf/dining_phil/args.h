// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <test/harness.h>

// Command line parametes
inline size_t HUNGER = 500;
inline size_t NUM_PHILOSOPHERS = 50;
inline bool OPTIMAL_ORDER = false;
inline size_t WORK_USEC = 1000;
inline bool MANUAL_LOCK_ORDER = false;

inline bool process_args(SystematicTestHarness& harness)
{
  HUNGER = harness.opt.is<size_t>("--hunger", HUNGER);
  NUM_PHILOSOPHERS =
    harness.opt.is<size_t>("--num_philosophers", NUM_PHILOSOPHERS);
  OPTIMAL_ORDER = harness.opt.has("--optimal_order");
  WORK_USEC = harness.opt.is<size_t>("--ponder_usec", 1000);

  MANUAL_LOCK_ORDER = harness.opt.has("--manual_lock_order");

  if (NUM_PHILOSOPHERS < 2)
  {
    std::cerr << "--num_philosophers must be at least 2" << std::endl;
    return false;
  }

  return true;
}