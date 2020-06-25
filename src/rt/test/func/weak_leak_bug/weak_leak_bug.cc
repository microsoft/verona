// Copyright Microsoft and Project Verona Contributors
// SPDX-License-Identifier: MIT

// See issue #84 for origin of this test.
#include <test/harness.h>

struct MyCown : VCown<MyCown>
{};

struct Msg : VAction<Msg>
{
  MyCown* m;
  Msg(MyCown* m) : m(m) {}

  void f()
  {
    Systematic::cout() << "Msg on " << m << std::endl;
  }
};

void run_test()
{
  MyCown* t = new MyCown;
  // HERE: the weak RC is never released.
  t->weak_acquire();

  Cown::schedule<Msg, YesTransfer>(t, t);
}

int main(int argc, char** argv)
{
  SystematicTestHarness h(argc, argv);

  Scheduler::set_detect_leaks(false);
  h.detect_leaks = false;

  h.run(run_test);
}
