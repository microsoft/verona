// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <test/harness.h>

inline uint32_t GENERATOR_COUNT = 2;
inline uint64_t ACCOUNTS_COUNT = 1000;
inline uint8_t ACCOUNT_EXTRA = 10;
inline uint32_t PER_GEN_TX_COUNT = 1'000'000;
inline uint32_t TX_BATCH = 10;

inline bool process_args(SystematicTestHarness& harness)
{
  GENERATOR_COUNT = harness.opt.is<uint32_t>("--generators", GENERATOR_COUNT);
  ACCOUNTS_COUNT = harness.opt.is<uint64_t>("--accounts", ACCOUNTS_COUNT);
  ACCOUNT_EXTRA = harness.opt.is<uint8_t>("--accounts_extra", ACCOUNT_EXTRA);
  PER_GEN_TX_COUNT = harness.opt.is<uint32_t>("--tx_count", PER_GEN_TX_COUNT);
  TX_BATCH = harness.opt.is<uint32_t>("--tx_batch", TX_BATCH);
  return true;
}
