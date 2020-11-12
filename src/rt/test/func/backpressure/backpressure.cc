// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "deadlock.h"
#include "fanin.h"
#include "unblock.h"

#include <test/harness.h>

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);
  harness.run(backpressure::fanin::test);
  harness.run(backpressure::deadlock::test);
  harness.run(backpressure::unblock::test);
  return 0;
}
