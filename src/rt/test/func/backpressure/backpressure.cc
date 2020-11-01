// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "deadlock.h"
#include "unblock.h"

#include <test/harness.h>

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);
  harness.run(backpressure_deadlock::test);
  harness.run(backpressure_unblock::test);
  return 0;
}
