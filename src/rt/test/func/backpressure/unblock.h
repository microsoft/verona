// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

/**
 * This test creates a scenario with two pairs of senders and receivers, where
 * the sender may overload the receiver and become muted. Another behaviour is
 * scheduled, requiring the sender from the first pair and the receiver from the
 * second pair. The backpressure system must ensure progress for the second
 * receiver, even if it is blocked on the first sender.
 */

/*
class C {}
main()
{
  var sender1 = cown.create(new C);
  var sender2 = cown.create(new C);
  var receiver1 = cown.create(new C);
  var receiver2 = cown.create(new C);

  when () { overload(sender1, receiver1) };
  when () { overload(sender2, receiver2) };
  when (sender1, receiver2) {};
}

overload(sender: C & imm, receiver: C & imm)
{
  var i: USize = 100;
  while (i > 0)
  {
    i = i - 1;
    when (sender) { when (receiver) {} }
  }
}
*/

#include "../../../verona.h"

#include <functional>

namespace backpressure_unblock
{
  using namespace verona::rt;

  struct C : public VCown<C>
  {};

  struct B : public VBehaviour<B>
  {
    std::function<void()> closure;

    B(std::function<void()> closure_) : closure(closure_) {}

    void f()
    {
      closure();
    }
  };

  void test()
  {
    auto* alloc = ThreadAlloc::get();
    auto* sender1 = new (alloc) C();
    auto* sender2 = new (alloc) C();
    auto* receiver1 = new (alloc) C();
    auto* receiver2 = new (alloc) C();

    Cown::acquire(sender1);
    Cown::acquire(receiver2);

    auto overload = [](Cown* sender, Cown* receiver) {
      for (size_t i = 0; i < 100; i++)
      {
        Cown::acquire(receiver);
        Cown::schedule<B>(sender, [receiver] {
          Cown::schedule<B, YesTransfer>(receiver, [] {});
        });
      }
      Cown::release(ThreadAlloc::get_noncachable(), sender);
      Cown::release(ThreadAlloc::get_noncachable(), receiver);
    };

    auto* anon1 = new (alloc) C();
    Cown::schedule<B, YesTransfer>(
      anon1, [=] { overload(sender1, receiver1); });
    auto* anon2 = new (alloc) C();
    Cown::schedule<B, YesTransfer>(
      anon2, [=] { overload(sender2, receiver2); });

    Cown* receivers[2] = {sender1, receiver2};
    Cown::schedule<B, YesTransfer>(2, receivers, [] {});
  }
}
