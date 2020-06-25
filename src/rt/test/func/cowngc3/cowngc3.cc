// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
/**
 * This examples is exploiting a complexity in the leak detector and
 * noticeboards.
 *
 * We have three initial Cowns
 *
 * Cown1 has a reference to Cown2
 *
 * Cown2 has a noticeboard allocates a fresh Cown, Cown4, and places it into the
 * noticeboard.
 *
 * Cown3 requests a leak detection is started.
 *
 * Cown1, observes the noticeboard. And sends its self reference to Cown4.
 *
 * If this occurs when Cown2 is initially Scan and Cown1 is PreScan, then the
 * system needs to ensure that it still scans the message sent to Cown4.  Hence,
 * freshly allocated Cowns, while PreScan cannot be made in the current LD
 * epoch.
 **/

/**
 * This example is using races outside Verona, these races cannot be correctly
 * memory managed without using the noticeboard implementation.  Until that
 * exists we must disable ASAN leak detection.
 **/
#if defined(__has_feature)
#  if __has_feature(address_sanitizer)
extern "C" const char* __asan_default_options()
{
  return "detect_leaks=0";
}
#  endif
#endif

#include <test/harness.h>

struct MyCown : public VCown<MyCown>
{
  MyCown() {}

  void trace(ObjectStack&) {}
};

/**
 * Faking up a noticeboard for this test.  Current functionality does not
 * support directly placing a Cown in a noticeboard.
 *
 * TODO Actually use noticeboard here when functionality implemented.
 **/
std::atomic<MyCown*> fake_noticeboard;

struct Ping : public VAction<Ping>
{
  Cown* c;
  Ping(Cown* c) : c(c) {}

  void f()
  {
    Systematic::cout() << "Ping on " << c << std::endl;
  }

  void trace(ObjectStack&) const {}
};

void noise()
{
  for (int n = 0; n < 3; n++)
  {
    // Make some noise
    auto c = new MyCown;
    Cown::schedule<Ping>(c, c);
  }
}

struct WantLD : public VAction<WantLD>
{
  Cown* c;
  WantLD(Cown* c) : c(c) {}

  void f()
  {
    Systematic::cout() << "WantLD on " << c << std::endl;
    Scheduler::want_ld();
  }
};

struct M0 : public VAction<M0>
{
  Cown* c;
  M0(Cown* c) : c(c) {}

  void f()
  {
    Systematic::cout() << "M0 on " << c << std::endl;

    // Allocate a new Cown, and put in noticebaord
    auto n = new MyCown;
    Systematic::cout() << "c4 = " << n << std::endl;

    yield();
    fake_noticeboard = n;
    yield();
  }
};

struct M2 : public VAction<M2>
{
  Cown* c;
  MyCown* keep_alive;
  int count_down;

  M2(Cown* c, MyCown* k, int count_down)
  : c(c), keep_alive(k), count_down(count_down)
  {}

  void f()
  {
    if (count_down == 0)
    {
      Systematic::cout() << "Fin M2 on " << c << " sending to " << keep_alive
                         << std::endl;
      Cown::schedule<Ping>(keep_alive, keep_alive);
    }
    else
    {
      Systematic::cout() << "Loop M2 on " << c << std::endl;
      Cown::schedule<M2>(c, c, keep_alive, count_down - 1);
    }
  }

  void trace(ObjectStack& ob) const
  {
    if (keep_alive != nullptr)
      ob.push(keep_alive);
  }
};

struct M1 : public VAction<M1>
{
  MyCown* m;
  M1(MyCown* m) : m(m) {}

  void f()
  {
    Systematic::cout() << "M1 on " << m << std::endl;

    MyCown* o;

    do
    {
      yield();
      o = fake_noticeboard;
    } while (o == nullptr);

    noise();

    Cown::schedule<Ping>(o, o);
    Cown::schedule<Ping>(o, o);
    Cown::schedule<Ping>(o, o);

    yield();

    Systematic::cout() << "Sending M2 to " << o << "with " << m << std::endl;
    Cown::schedule<M2>(o, o, m, 20);

    yield();
    noise();
  }
};

void run_test()
{
  fake_noticeboard = nullptr;
  auto c1 = new MyCown;
  auto c2 = new MyCown;
  auto c3 = new MyCown;
  Systematic::cout() << "c1 = " << c1 << std::endl;
  Systematic::cout() << "c2 = " << c2 << std::endl;
  Systematic::cout() << "c3 = " << c3 << std::endl;

  Cown::schedule<Ping>(c1, c1);
  Cown::schedule<Ping>(c2, c2);
  Cown::schedule<WantLD>(c3, c3);
  Cown::schedule<M1>(c1, c1);
  Cown::schedule<M0>(c2, c2);
  Cown::schedule<Ping>(c1, c1);
  Cown::schedule<Ping>(c2, c2);

  noise();
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  if (harness.detect_leaks)
    std::cout << "This example leaks, run with '--allow_leaks'" << std::endl;
  else
    harness.run(run_test);
  return 0;
}
