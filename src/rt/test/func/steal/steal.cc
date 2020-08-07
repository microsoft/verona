// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <ctime>
#include <test/harness.h>

struct Runner : public VCown<Runner>
{
  Runner() {}
};

void schedule_run(size_t decay);

struct Run : public VBehaviour<Run>
{
  Runner* r;
  size_t decay;

  Run(Runner* r, size_t decay) : r(r), decay(decay) {}

  void f()
  {
    schedule_run(decay);
  }
};

void schedule_run(size_t decay)
{
  if (decay == 0)
    return;

  auto* alloc = ThreadAlloc::get();
  auto runner = new Runner();
  Cown::schedule<Run>(runner, runner, decay - 1);
  Cown::release(alloc, runner);
}

void basic_test(size_t cores)
{
  // There should be one fewer runners than cores to cause
  // stealing to occur a lot.
  for (size_t i = 0; i < cores - 1; i++)
  {
    schedule_run(3);
  }
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);
  harness.run(basic_test, harness.cores);
  return 0;
}
