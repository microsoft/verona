// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <test/harness.h>

struct CCown : public VCown<CCown>
{
  CCown* child;
  CCown(CCown* child_) : child(child_) {}

  void trace(ObjectStack& fields) const
  {
    if (child != nullptr)
      fields.push(child);
  }
};

struct Nop : public VBehaviour<Nop>
{
  Nop() {}

  void f() {}
};

struct Mess : public VBehaviour<Mess>
{
  CCown* ccown;
  size_t timer;

  Mess(CCown* ccown, size_t timer_) : ccown(ccown), timer(timer_) {}

  void f()
  {
    if (timer == 0)
    {
      if (ccown->child != nullptr)
      {
        Systematic::cout() << "Child message: " << ccown->child << std::endl;
        Cown::schedule<Mess>(ccown->child, ccown->child, (size_t)0);
      }
      return;
    }

    if (timer == 30)
    {
      Scheduler::want_ld();
    }

    Systematic::cout() << "Self: " << timer << std::endl;
    Cown::schedule<Mess>(ccown, ccown, timer - 1);
  }
};

struct Go : public VBehaviour<Go>
{
  CCown* start;

  Go(CCown* start) : start(start) {}

  void f()
  {
    Cown::schedule<Mess>(start, start, (size_t)31);
    cown::release(ThreadAlloc::get(), start);
  }
};

void test_cown_gc()
{
  auto L = new CCown(nullptr);
  Systematic::cout() << "L:" << L << std::endl;
  auto M = new CCown(L);
  Systematic::cout() << "M:" << M << std::endl;
  auto S = new CCown(M);
  Systematic::cout() << "S:" << S << std::endl;
  Cown::schedule<Go>(L, S);
  Cown::schedule<Nop>(M);
}

/**
 * This is a test that captures the following interesting behaviour
 *
 * There are three cowns L, M, S
 *  Main - Go(S) -> L  (1)
 *  Main - Nop -> M (2)
 *
 *  L.Go(S)    -Mess(31)-> S    (after 1)
 *  S.Mess(31) -Mess(30)-> S
 *  S.Mess(30) WantLD Signalled
 *  S.Mess(30) -Mess(29)-> S
 *  ...
 *  S.Mess(0)  -Mess(0)->  M    (after 2)
 *  M.Mess(0)  -Mess(0)->  L
 *
 * The aim of this test is for L to become descheduled at the same time as
 * S is signalling it wants the LD.  This triggers many awkward cases in the
 * Cown collection algorithm.
 */
int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  harness.run(test_cown_gc);

  return 0;
}
