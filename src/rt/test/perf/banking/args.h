// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <test/harness.h>

// Command line parametes
inline size_t TRANSACTIONS = 50;
inline size_t NUM_WORKERS = 36;
inline size_t NUM_ACCOUNTS = 36;
inline size_t WORK_USEC = 1000;

inline bool process_args(SystematicTestHarness& harness)
{
  TRANSACTIONS = harness.opt.is<size_t>("--num_trans", TRANSACTIONS);
  NUM_WORKERS = harness.opt.is<size_t>("--cores", NUM_WORKERS);
  WORK_USEC = harness.opt.is<size_t>("--work_usec", WORK_USEC);
  NUM_ACCOUNTS = harness.opt.is<size_t>("--accounts", NUM_ACCOUNTS);

  return true;
}