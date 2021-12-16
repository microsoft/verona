// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <random>
#include <test/harness.h>
#include <test/xoroshiro.h>

/**
 * This tests the cown leak detector.
 *
 *
 * First test:
 * (TODO: want it to fail if cown::scan_stack is disabled)
 *
 * Creates a ring of RCowns, each with a number of child CCowns. The child
 * CCowns are reachable via:
 *   - a member array
 *   - a pointer to a RegionTrace
 *   - a pointer to a RegionArena
 *   - two pointers to two immutables
 *
 * (More complicated structures are tested in cowngc4.)
 *
 * The initialization for each of case can be commented out for debugging.
 *
 * Each RCown creates an "grandchild" CCown, `shared_child`, that is shared by
 * its child CCowns.
 *
 * The test starts by sending a Ping to the first RCown. If its forward count
 * is nonzero, it sends a Ping to the next RCown in the ring, a Pong to each of
 * its child CCowns, and then decrements its forward count. After 3/4 of the
 * forward count has occurred, we drop a bunch of child cowns.
 *
 * When a CCown (that is not a shared child) receives a Pong, it sends multiple
 * Pongs to its shared child. The shared child requests an LD run.
 *
 * We expect the LD to properly handle the cowns, shared cowns, and all
 * messages, including in-flight messages.
 *
 *
 * Second test:
 * Check that the LD runs even if no work is scheduled.
 **/

struct PRNG
{
#ifdef USE_SYSTEMATIC_TESTING
  // Use xoroshiro for systematic testing, because it's simple and
  // and deterministic across platforms.
  xoroshiro::p128r64 rand;
#else
  // We don't mind data races for our PRNG, because concurrent testing means
  // our results will already be nondeterministic. However, data races may
  // cause xoroshiro to abort.
  std::mt19937_64 rand;
#endif

  PRNG(size_t seed) : rand(seed) {}

  uint64_t next()
  {
#ifdef USE_SYSTEMATIC_TESTING
    return rand.next();
#else
    return rand();
#endif
  }

  void seed(size_t seed)
  {
#ifdef USE_SYSTEMATIC_TESTING
    return rand.set_state(seed);
#else
    return rand.seed(seed);
#endif
  }
};

static constexpr uint64_t others_count = 3;

struct CCown;
struct RCown;
static RCown* rcown_first;

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

struct O : public V<O>
{
  O* f1 = nullptr;
  CCown* cown = nullptr;

  void trace(ObjectStack& st) const
  {
    if (f1 != nullptr)
      st.push(f1);
    if (cown != nullptr)
      st.push(cown);
  }
};
// The types are used for documentation purposes only.
using OTrace = O;
using OArena = O;

struct RCown : public VCown<RCown>
{
  uint64_t forward;
  RCown* next; // never null after initialization

  CCown* array[others_count] = {}; // may contain null
  OTrace* otrace = nullptr; // may be null
  OArena* oarena = nullptr; // may be null
  OTrace* imm1 = nullptr; // may be null
  OTrace* imm2 = nullptr; // may be null

  RCown(size_t more, uint64_t forward_count) : forward(forward_count)
  {
    auto& alloc = ThreadAlloc::get();

    if (rcown_first == nullptr)
      rcown_first = this;

    Systematic::cout() << "Cown " << this << std::endl;

    auto shared_child = new CCown(nullptr);
    Systematic::cout() << "  shared " << shared_child << std::endl;

    // Initialize array
    {
      for (uint64_t i = 0; i < others_count; i++)
      {
        array[i] = new CCown(shared_child);
        Systematic::cout() << "  child " << array[i] << std::endl;
        Cown::acquire(shared_child); // acquire on behalf of child CCown
      }
    }

    // Initialize otrace
    {
      otrace = new (RegionType::Trace) OTrace;
      otrace->cown = new CCown(shared_child);
      Systematic::cout() << "  child " << otrace->cown << std::endl;
      // Transfer ownership of child CCown to the regions.
      RegionTrace::insert<TransferOwnership::YesTransfer>(
        alloc, otrace, otrace->cown);
      Cown::acquire(shared_child); // acquire on behalf of child CCown
    }

    // Initialize oarena
    {
      oarena = new (RegionType::Arena) OArena;
      oarena->cown = new CCown(shared_child);
      Systematic::cout() << "  child " << oarena->cown << std::endl;
      // Transfer ownership of child CCown to the regions.
      RegionArena::insert<TransferOwnership::YesTransfer>(
        alloc, oarena, oarena->cown);
      Cown::acquire(shared_child); // acquire on behalf of child CCown
    }

    // Initialize imm1 and imm2
    {
      // Create two immutables. Each is a two object cycle, but we pass a
      // different object to RCown, to get coverage of RC vs SCC objects.
      auto r1 = new (RegionType::Trace) OTrace;
      {
        UsingRegion ur(r1);
        r1->f1 = new OTrace;
        r1->f1->f1 = r1;
        r1->cown = new CCown(shared_child);
        Systematic::cout() << "  child " << r1->cown << std::endl;
        Cown::acquire(shared_child); // acquire on behalf of child CCown
        r1->f1->cown = new CCown(shared_child);
        Systematic::cout() << "  child " << r1->f1->cown << std::endl;
        Cown::acquire(shared_child); // acquire on behalf of child CCown
      }

      auto r2 = new (RegionType::Trace) OTrace;
      {
        UsingRegion ur(r2);
        r2->f1 = new OTrace;
        r2->f1->f1 = r2;
        r2->cown = new CCown(shared_child);
        Systematic::cout() << "  child " << r2->cown << std::endl;
        Cown::acquire(shared_child); // acquire on behalf of child CCown
        r2->f1->cown = new CCown(shared_child);
        Systematic::cout() << "  child " << r2->f1->cown << std::endl;
        Cown::acquire(shared_child); // acquire on behalf of child CCown
      }

      freeze(r1);
      freeze(r2);
      imm1 = r1;
      imm2 = r2->f1;

      // Release child CCowns that are now owned by the immutables.
      Cown::release(alloc, r1->cown);
      Cown::release(alloc, r1->f1->cown);
      Cown::release(alloc, r2->cown);
      Cown::release(alloc, r2->f1->cown);

      // Want to make sure one of the objects is RC and the other is SCC_PTR.
      check(imm1->debug_is_rc() || imm2->debug_is_rc());
      check(imm1->debug_is_rc() != imm2->debug_is_rc());
    }

    // Release our (RCown's) refcount on the shared_child.
    Cown::release(alloc, shared_child);

    if (more != 0)
      next = new RCown(more - 1, forward_count);
    else
      next = rcown_first;

    Systematic::cout() << "  next " << next << std::endl;
  }

  void trace(ObjectStack& fields) const
  {
    for (uint64_t i = 0; i < others_count; i++)
    {
      if (array[i] != nullptr)
        fields.push(array[i]);
    }

    if (otrace != nullptr)
      fields.push(otrace);

    if (oarena != nullptr)
      fields.push(oarena);

    if (imm1 != nullptr)
      fields.push(imm1);

    if (imm2 != nullptr)
      fields.push(imm2);

    check(next != nullptr);
    fields.push(next);
  }
};

struct Pong : public VBehaviour<Pong>
{
  CCown* ccown;
  Pong(CCown* ccown) : ccown(ccown) {}

  void f()
  {
    if (ccown->child != nullptr)
    {
      for (int n = 0; n < 20; n++)
        Cown::schedule<Pong>(ccown->child, ccown->child);
    }
    else
    {
      Scheduler::want_ld();
    }
  }
};

struct Ping : public VBehaviour<Ping>
{
  RCown* rcown;
  PRNG* rand;
  Ping(RCown* rcown, PRNG* rand) : rcown(rcown), rand(rand) {}

  void f()
  {
    if (rcown->forward > 0)
    {
      // Forward Ping to next RCown.
      Cown::schedule<Ping>(rcown->next, rcown->next, rand);

      // Send Pongs to child CCowns.
      for (uint64_t i = 0; i < others_count; i++)
      {
        if (rcown->array[i] != nullptr)
          Cown::schedule<Pong>(rcown->array[i], rcown->array[i]);
      }
      if (rcown->otrace != nullptr && rcown->otrace->cown != nullptr)
        Cown::schedule<Pong>(rcown->otrace->cown, rcown->otrace->cown);
      if (rcown->oarena != nullptr && rcown->oarena->cown != nullptr)
        Cown::schedule<Pong>(rcown->oarena->cown, rcown->oarena->cown);
      if (rcown->imm1 != nullptr)
      {
        auto c1 = rcown->imm1->cown;
        auto c2 = rcown->imm1->f1->cown;
        Cown::schedule<Pong>(c1, c1);
        Cown::schedule<Pong>(c2, c2);
      }
      if (rcown->imm2 != nullptr)
      {
        auto c1 = rcown->imm2->cown;
        auto c2 = rcown->imm2->f1->cown;
        Cown::schedule<Pong>(c1, c1);
        Cown::schedule<Pong>(c2, c2);
      }

      // Randomly introduce a few leaks. We don't want to do this for every
      // Ping, only about a quarter.
      switch (rand->next() % 12)
      {
        case 0:
        {
          size_t idx = rand->next() % others_count;
          if (rcown->array[idx] != nullptr)
          {
            Systematic::cout() << "RCown " << rcown << " is leaking cown "
                               << rcown->array[idx] << std::endl;
            // TODO: Sometimes the leak detector doesn't catch this. Although
            // the cown is leaked, it might still be scheduled, so it's treated
            // as live. For now, we'll explicitly release the cown.
            Cown::release(ThreadAlloc::get(), rcown->array[idx]);
            rcown->array[idx] = nullptr;
          }
          break;
        }
        case 1:
        {
          // Can't drop pointer to region, otherwise the region would leak.
          // Instead, we drop the pointer to the region's cown. We also need to
          // clear the remembered set.
          if (rcown->otrace != nullptr && rcown->otrace->cown != nullptr)
          {
            Systematic::cout() << "RCown " << rcown << " is leaking cown "
                               << rcown->otrace->cown << std::endl;
            auto* reg = RegionTrace::get(rcown->otrace);
            reg->discard(ThreadAlloc::get());
            rcown->otrace->cown = nullptr;
          }
          break;
        }
        case 2:
        {
          // Can't drop pointer to region, otherwise the region would leak.
          // Instead, we drop the pointer to the region's cown. We also need to
          // clear the remembered set.
          if (rcown->oarena != nullptr && rcown->oarena->cown != nullptr)
          {
            Systematic::cout() << "RCown " << rcown << " is leaking cown "
                               << rcown->oarena->cown << std::endl;
            auto* reg = RegionArena::get(rcown->oarena);
            reg->discard(ThreadAlloc::get());
            rcown->oarena->cown = nullptr;
          }
          break;
        }
        default:
          break;
      }

      rcown->forward--;
    }
    if (rcown->next == rcown_first)
    {
      Systematic::cout() << "Loop " << rcown->forward << std::endl;
      // Scheduler::want_ld();
    }
  }
};

void test_cown_gc(
  uint64_t forward_count,
  size_t ring_size,
  SystematicTestHarness* h,
  PRNG* rand)
{
  rcown_first = nullptr;
  auto a = new RCown(ring_size, forward_count);
  rand->seed(h->current_seed());
  Cown::schedule<Ping>(a, a, rand);
}

void test_cown_gc_before_sched()
{
  auto a = new CCown(nullptr);
  auto& alloc = ThreadAlloc::get();
  Cown::release(alloc, a);
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);
  PRNG rand(harness.seed_lower);

  size_t ring = harness.opt.is<size_t>("--ring", 10);
  uint64_t forward = harness.opt.is<uint64_t>("--forward", 10);

  harness.run(test_cown_gc, forward, ring, &harness, &rand);
  harness.run(test_cown_gc_before_sched);

  return 0;
}
