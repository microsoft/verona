// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <test/harness.h>
// Harness must come before tests.
#include "./notify_basic.h"
#include "./notify_coalesce.h"
#include "./notify_interleave.h"

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  harness.run(notify_basic::basic_test);

  harness.run(notify_interleave::run_test);

  // Here we ensure single-core so that we can assert the number of times
  // `notified` is called.
  if (harness.cores == 1)
    harness.run(notify_coalesce::run_test);

  return 0;
}
