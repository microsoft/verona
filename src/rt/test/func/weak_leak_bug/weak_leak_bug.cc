// Copyright Microsoft and Project Verona Contributors
// SPDX-License-Identifier: MIT

// See issue #84 for origin of this test.
// This is probably no longer relevant as the teardown without the
// leak detector is much simpler.  Leaving the test incase that
// changes.

#include <test/harness.h>

#if defined(__has_feature)
#  if __has_feature(address_sanitizer)
extern "C" const char* __asan_default_options()
{
  return "detect_leaks=0";
}
#  endif
#endif

struct MyCown : VCown<MyCown>
{};

struct Msg : VBehaviour<Msg>
{
  MyCown* m;
  Msg(MyCown* m) : m(m) {}

  void f()
  {
    Logging::cout() << "Msg on " << m << std::endl;
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
