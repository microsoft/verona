// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

/**
 * This test creates a possible deadlock between cowns c1, c2, and c3 where the
 * acquisition order of these cowns is c1 before c2 before c3. The following
 * order of events may occur:
 *  1. c1 is overloaded
 *  2. c3 creates a behaviour {c1}. c1 mutes c3.
 *  3. c2 creates a behaviour {c2, c3}. c2 is acquired and blocked on c3 until
 *     c3 is unmuted and c3 runs this behaviour.
 *  4. c1 creates a behaviour {c1, c2}. c1 is acquired and blocked on c2 until
 *     c2 is rescheduled and runs this behaviour. The priority of c2 is raised.
 *
 * In this scenario, it is possible for all three cowns to be deadlocked, unless
 * c2 is unblocked when its priority is raised by unmuting c1.
 */

/*
class C {}
class Main
{
  main()
  {
    var c1 = cown.create(new C);
    var c2 = cown.create(new C);
    var c3 = cown.create(new C);
    // overload c1
    var i = 100;
    while i > 0
    {
      i = i - 1;
      when (c1) {};
    };
    // c3 send {c1}
    when (var _ = c3) { when (c1) {} };
    // c2 send {c2, c3}
    when (var _ = c2) { when (c2, c3) {} };
    // c1 send {c1, c2}
    when (var _ = c1) { when (c1, c2) {} }
  }
}
*/

#include "../../../verona.h"

namespace backpressure_deadlock
{
  using namespace verona::rt;

  struct C : public VCown<C>
  {};

  struct B : public VBehaviour<B>
  {
    std::vector<C*> receivers;

    B(std::vector<C*> receivers_) : receivers(receivers_) {}

    void f()
    {
      if (!receivers.empty())
      {
        Cown::schedule<B, YesTransfer>(
          receivers.size(), (Cown**)receivers.data(), std::vector<C*>{});
      }
    }
  };

  void test()
  {
    auto* alloc = ThreadAlloc::get();
    auto* c1 = new (alloc) C();
    auto* c2 = new (alloc) C();
    auto* c3 = new (alloc) C();

    // overload c1
    for (size_t i = 0; i < 100; i++)
      Cown::schedule<B, NoTransfer, std::vector<C*>>(c1, {});

    // c3 send {c1}
    Cown::acquire(c1);
    Cown::schedule<B, YesTransfer, std::vector<C*>>(c3, {c1});
    // c2 send {c2, c3}
    Cown::acquire(c2);
    Cown::acquire(c3);
    Cown::schedule<B, YesTransfer, std::vector<C*>>(c2, {c2, c3});
    // c1 send {c1, c2}
    Cown::acquire(c1);
    Cown::acquire(c2);
    Cown::schedule<B, YesTransfer, std::vector<C*>>(c1, {c1, c2});
  }
}
