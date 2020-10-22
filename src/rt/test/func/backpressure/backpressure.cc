// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <test/harness.h>

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

void deadlock()
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

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);
  harness.run(deadlock);
  return 0;
}
