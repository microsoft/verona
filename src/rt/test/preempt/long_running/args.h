#pragma once

#include <test/harness.h>

// Command line parameters.

inline size_t NUM_SMALL = 500;

inline bool process_args(SystematicTestHarness& harness)
{
  NUM_SMALL = harness.opt.is<size_t>("--num_small", NUM_SMALL);
  return true;
}
