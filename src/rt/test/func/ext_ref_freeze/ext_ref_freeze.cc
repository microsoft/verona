// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <test/harness.h>
#include <verona.h>

using namespace snmalloc;
using namespace verona::rt;

// This only tests trace regions.
// At the moment, it does not make sense to freeze arena regions.

struct B : public VCown<B>
{};

struct C : public V<C>
{
  C* f1 = nullptr;
  B* b = nullptr;
  void trace(ObjectStack& st) const
  {
    if (f1 != nullptr)
      st.push(f1);
    if (b != nullptr)
      st.push(b);
  }
};

struct Ping : public VBehaviour<Ping>
{
  void f() {}
};

enum Phase
{
  SETUP,
  REUSE,
  EXIT,
};

struct A : public VCown<A>
{
  Phase state = SETUP;
  C* r = nullptr;

  void trace(ObjectStack& st) const
  {
    if (r != nullptr)
      st.push(r);
  }
};

ExternalRef* g_ext_ref = nullptr;

struct Loop : public VBehaviour<Loop>
{
  A* a;
  Loop(A* a) : a(a) {}

  void f()
  {
    auto& state = a->state;

    auto& alloc = ThreadAlloc::get();
    (void)alloc;
    switch (state)
    {
      case SETUP:
      {
        auto r = new (RegionType::Trace) C;

        {
          UsingRegion ur(r);
          r->f1 = new C;
          r->f1->f1 = new C;
          r->f1->f1->f1 = r->f1;
          r->f1->b = new B;
        }

        a->r = r;
        RegionTrace::insert<YesTransfer>(alloc, a->r, a->r->f1->b);
        {
          UsingRegion ur(a->r);
          g_ext_ref = create_external_reference(a->r->f1);
        }
        freeze(a->r);
        state = REUSE;
        Cown::schedule<Loop>(a, a);
        return;
      }
      case REUSE:
      {
        state = EXIT;
        Cown::schedule<Ping>(a->r->f1->b);
        Cown::schedule<Loop>(a, a);
        return;
      }
      case EXIT:
      {
        Immutable::release(alloc, g_ext_ref);
        g_ext_ref = nullptr;
        return;
      }
      default:
        abort();
    }
  }
};

// This test confirms that the bottom bit marked for `has_ext_ref` is cleared
// when the belonging region is frozen.
// The overall procedure consists of five steps, as shown in enum `Phase`:
// 1. Advancing the global epoch so that it matches the epoch corresponding to
//    the one with `has_ext_ref` set.
// 2. Setting up the region and its internal connections, creating a ext_ref to
//    an internal node, attach a cown to that internal node, and freeze the
//    region.
// 3. Sending a message to the cown in step 2. Should fail if the cown was
//    prematurely collected because the `has_ext_ref` bit confused the tracing
//    algorithm.
// 4. Clean up resource, and exit.
void run_test()
{
  auto& alloc = ThreadAlloc::get();
  auto a = new A;
  Cown::schedule<Loop>(a, a);
  Cown::release(alloc, a);
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  harness.run(run_test);

  return 0;
}
