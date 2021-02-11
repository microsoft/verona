// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <random>
#include <test/harness.h>
#include <test/xoroshiro.h>

/**
 * This tests the cown leak detector. This is a variant of cowngc1.
 * (TODO: want it to fail if cown::scan_stack is disabled)
 *
 * Creates a ring of RCowns, each with child regions. Child CCowns are
 * reachable from within the region. We test the following cases:
 *   - region with an object graph
 *   - region with subregions
 *   - region with immutables
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

template<RegionType region_type>
struct RCown;

// We'll need to do some ugly casting later on...
static RCown<RegionType::Trace>* rcown_first;

template<RegionType region_type>
struct O : public V<O<region_type>, region_type>
{
  O<region_type>* f = nullptr;
  O<RegionType::Trace>* f1 = nullptr;
  O<RegionType::Arena>* f2 = nullptr;
  O<RegionType::Trace>* imm1 = nullptr;
  O<RegionType::Trace>* imm2 = nullptr;
  CCown* cown = nullptr;

  void trace(ObjectStack& st) const
  {
    if (f != nullptr)
      st.push(f);
    if (f1 != nullptr)
      st.push(f1);
    if (f2 != nullptr)
      st.push(f2);
    if (imm1 != nullptr)
      st.push(imm1);
    if (imm2 != nullptr)
      st.push(imm2);
    if (cown != nullptr)
      st.push(cown);
  }

  void finaliser(Object* region, ObjectStack& sub_regions)
  {
    Object::add_sub_region(f1, region, sub_regions);
    Object::add_sub_region(f2, region, sub_regions);
  }
};

using OTrace = O<RegionType::Trace>;
using OArena = O<RegionType::Arena>;

template<RegionType region_type>
struct RCown : public VCown<RCown<region_type>>
{
  using RegionClass = typename RegionType_to_class<region_type>::T;
  using Self = RCown<region_type>;
  using Reg = O<region_type>;

  uint64_t forward;
  uint64_t threshold;
  Self* next; // never null after initialization

  Reg* reg_with_graph = nullptr; // may be null
  Reg* reg_with_sub = nullptr; // may be null
  Reg* reg_with_imm = nullptr; // may be null

  RCown(size_t more, uint64_t forward_count)
  : forward(forward_count), threshold(forward_count / 4)
  {
    auto* alloc = ThreadAlloc::get();

    if (rcown_first == nullptr)
      rcown_first = (RCown<RegionType::Trace>*)this;

    Systematic::cout() << "Cown " << this << std::endl;

    auto shared_child = new CCown(nullptr);
    Systematic::cout() << "  shared " << shared_child << std::endl;

    // Initialize region with object graph. We'll make a short linked list.
    {
      auto* r = new Reg;
      r->f = new (r) Reg;
      r->f->f = new (r) Reg;
      r->f->f->f = r;

      // Construct a CCown and give it to the region.
      auto c = new CCown(shared_child);
      Systematic::cout() << "  child " << c << std::endl;
      RegionClass::template insert<TransferOwnership::YesTransfer>(alloc, r, c);
      Cown::acquire(shared_child); // acquire on behalf of child CCown

      reg_with_graph = r;
      reg_with_graph->f->f->cown = c;
    }

    // Initialize a linked list of regions.
    {
      auto* r = new Reg;
      r->f1 = new OTrace;
      r->f1->f2 = new OArena;
      r->f1->f2->f2 = new OArena;

      // Construct a CCown and give it to the last region.
      auto c = new CCown(shared_child);
      Systematic::cout() << "  child " << c << std::endl;
      RegionArena::insert<TransferOwnership::YesTransfer>(
        alloc, r->f1->f2->f2, c);
      Cown::acquire(shared_child); // acquire on behalf of child CCown

      reg_with_sub = r;
      reg_with_sub->f1->f2->f2->cown = c;
    }

    // Initialize region with immutables.
    {
      reg_with_imm = new Reg;

      // Create two immutables. Each is a two object cycle, but we pass a
      // different object to reg_with_imm, to get coverage of RC vs SCC
      // objects.
      auto r1 = new OTrace;
      r1->f1 = new (r1) OTrace;
      r1->f1->f1 = r1;
      r1->cown = new CCown(shared_child);
      Systematic::cout() << "  child " << r1->cown << std::endl;
      Cown::acquire(shared_child); // acquire on behalf of child CCown
      r1->f1->cown = new CCown(shared_child);
      Systematic::cout() << "  child " << r1->f1->cown << std::endl;
      Cown::acquire(shared_child); // acquire on behalf of child CCown

      auto r2 = new OTrace;
      r2->f1 = new (r2) OTrace;
      r2->f1->f1 = r2;
      r2->cown = new CCown(shared_child);
      Systematic::cout() << "  child " << r2->cown << std::endl;
      Cown::acquire(shared_child); // acquire on behalf of child CCown
      r2->f1->cown = new CCown(shared_child);
      Systematic::cout() << "  child " << r2->f1->cown << std::endl;
      Cown::acquire(shared_child); // acquire on behalf of child CCown

      Freeze::apply(alloc, r1);
      Freeze::apply(alloc, r2);
      reg_with_imm->imm1 = r1;
      reg_with_imm->imm2 = r2->f1;

      // Transfer ownership of immutables to the region.
      RegionClass::template insert<TransferOwnership::YesTransfer>(
        alloc, reg_with_imm, r1);
      RegionClass::template insert<TransferOwnership::YesTransfer>(
        alloc, reg_with_imm, r2);

      // Release child CCowns that are now owned by the immutables.
      Cown::release(alloc, r1->cown);
      Cown::release(alloc, r1->f1->cown);
      Cown::release(alloc, r2->cown);
      Cown::release(alloc, r2->f1->cown);

      // Want to make sure one of the objects is RC and the other is SCC_PTR.
      check(
        reg_with_imm->imm1->debug_is_rc() || reg_with_imm->imm2->debug_is_rc());
      check(
        reg_with_imm->imm1->debug_is_rc() != reg_with_imm->imm2->debug_is_rc());
    }

    // Release our (RCown's) refcount on the shared_child.
    Cown::release(alloc, shared_child);

    if (more != 0)
      next = new Self(more - 1, forward_count);
    else
      next = (Self*)rcown_first;

    Systematic::cout() << "  next " << next << std::endl;
  }

  void trace(ObjectStack& fields) const
  {
    if (reg_with_graph != nullptr)
      fields.push(reg_with_graph);

    if (reg_with_sub != nullptr)
      fields.push(reg_with_sub);

    if (reg_with_imm != nullptr)
      fields.push(reg_with_imm);

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

template<RegionType region_type>
struct Ping : public VBehaviour<Ping<region_type>>
{
  using RegionClass = typename RegionType_to_class<region_type>::T;

  RCown<region_type>* rcown;
  PRNG* rand;
  Ping(RCown<region_type>* rcown, PRNG* rand) : rcown(rcown), rand(rand) {}

  void f()
  {
    if (rcown->forward > 0)
    {
      // Forward Ping to next RCown.
      Cown::schedule<Ping<region_type>>(rcown->next, rcown->next, rand);

      // Send Pongs to child CCowns.
      if (
        rcown->reg_with_graph != nullptr &&
        rcown->reg_with_graph->f->f->cown != nullptr)
      {
        auto c = rcown->reg_with_graph->f->f->cown;
        Cown::schedule<Pong>(c, c);
      }
      if (
        rcown->reg_with_sub != nullptr &&
        rcown->reg_with_sub->f1->f2->f2->cown != nullptr)
      {
        auto c = rcown->reg_with_sub->f1->f2->f2->cown;
        Cown::schedule<Pong>(c, c);
      }
      if (rcown->reg_with_imm != nullptr)
      {
        auto c1 = rcown->reg_with_imm->imm1->cown;
        auto c2 = rcown->reg_with_imm->imm1->f1->cown;
        Cown::schedule<Pong>(c1, c1);
        Cown::schedule<Pong>(c2, c2);
        c1 = rcown->reg_with_imm->imm2->cown;
        c2 = rcown->reg_with_imm->imm2->f1->cown;
        Cown::schedule<Pong>(c1, c1);
        Cown::schedule<Pong>(c2, c2);
      }

      // Randomly introduce a few leaks. We don't want to do this for every
      // Ping, only about a quarter.
      switch (rand->next() % 8)
      {
        case 0:
        {
          // Can't drop pointer to region, otherwise the region would leak.
          // Instead, we drop the pointer to the region's cown. We also need to
          // clear the remembered set.
          if (
            rcown->reg_with_graph != nullptr &&
            rcown->reg_with_graph->f->f->cown != nullptr)
          {
            Systematic::cout()
              << "RCown " << rcown << " is leaking cown "
              << rcown->reg_with_graph->f->f->cown << std::endl;
            auto* reg = RegionClass::get(rcown->reg_with_graph);
            reg->discard(ThreadAlloc::get());
            rcown->reg_with_graph->f->f->cown = nullptr;
          }
          break;
        }
        case 1:
        {
          // Can't drop pointer to region, otherwise the region would leak.
          // Instead, we drop the pointer to the region's cown. We also need to
          // clear the remembered set.
          if (
            rcown->reg_with_sub != nullptr &&
            rcown->reg_with_sub->f1->f2->f2 != nullptr)
          {
            Systematic::cout() << "RCown " << rcown << " is leaking cown "
                               << rcown->reg_with_sub->f1->f2->f2 << std::endl;
            auto* reg = RegionArena::get(rcown->reg_with_sub->f1->f2->f2);
            reg->discard(ThreadAlloc::get());
            rcown->reg_with_sub->f1->f2->f2->cown = nullptr;
          }
          break;
        }
        default:
          break;
      }

      rcown->forward--;
    }
    if (rcown->next == (RCown<region_type>*)rcown_first)
    {
      Systematic::cout() << "Loop " << rcown->forward << std::endl;
      // Scheduler::want_ld();
    }
  }
};

template<RegionType region_type>
void test_cown_gc(
  uint64_t forward_count,
  size_t ring_size,
  SystematicTestHarness* h,
  PRNG* rand)
{
  rcown_first = nullptr;
  auto a = new RCown<region_type>(ring_size, forward_count);
  rand->seed(h->current_seed());
  Cown::schedule<Ping<region_type>>(a, a, rand);
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);
  PRNG rand(harness.seed_lower);

  size_t ring = harness.opt.is<size_t>("--ring", 10);
  uint64_t forward = harness.opt.is<uint64_t>("--forward", 10);

  harness.run(test_cown_gc<RegionType::Trace>, forward, ring, &harness, &rand);
  harness.run(test_cown_gc<RegionType::Arena>, forward, ring, &harness, &rand);

  return 0;
}
