// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <test/harness.h>
#include <test/log.h>
#include <verona.h>

using namespace snmalloc;
using namespace verona::rt;

static std::atomic<int> live_count;

template<RegionType region_type>
struct C1 : public V<C1<region_type>, region_type>
{
  C1<region_type>* f1 = nullptr;
  C1<region_type>* f2 = nullptr;

  void trace(ObjectStack& st) const
  {
    if (f1 != nullptr)
      st.push(f1);

    if (f2 != nullptr)
      st.push(f2);
  }
};

template<RegionType region_type>
class C2 : public V<C2<region_type>, region_type>
{
public:
  C2<region_type>* f1 = nullptr;
  enum State
  {
    LIVE,
    FINALISED,
    DESTRUCTED
  };

  State state;

  C2() : state(LIVE) {}

  void trace(ObjectStack& st) const
  {
    // Tracing should never happen after destruction
    check(state == LIVE || state == FINALISED);

    if (f1 != nullptr)
      st.push(f1);
  }

  void finaliser(Object* region, ObjectStack& sub_regions)
  {
    check(state == LIVE);
    state = FINALISED;
    Object::add_sub_region(f1, region, sub_regions);
  }

  ~C2()
  {
    check(state == FINALISED);
    state = DESTRUCTED;
  }
};

template<RegionType region_type>
class F1 : public V<F1<region_type>, region_type>
{
public:
  F1()
  {
    live_count++;
  }

  void finaliser(Object*, ObjectStack&)
  {
    live_count--;
    logger::cout() << "Finalised" << std::endl;
  }
};

template<RegionType region_type>
class F2 : public V<F2<region_type>, region_type>
{
public:
  int id;
  F2* parent;
  F2* child;

  // Force RegionArena to allocate this object in the large object ring, where
  // we're more likely to see dangling pointers.
  uint8_t data[1024 * 1024 * 2];

  F2(int id, F2* parent) : id(id), parent(parent), child(nullptr)
  {
    live_count++;
  }

  void finaliser(Object*, ObjectStack&)
  {
    live_count--;
    logger::cout() << "Finalised: " << id << std::endl;
    logger::cout() << "Visiting:";
    visit();
    logger::cout() << std::endl;
  }

  // Quick and dirty check that pointers aren't dangling.
  void visit()
  {
    logger::cout() << " " << id;

    F2* p = parent;
    while (p != nullptr)
    {
      logger::cout() << " " << p->id;
      p = p->parent;
    }

    p = child;
    while (p != nullptr)
    {
      logger::cout() << " " << p->id;
      p = p->child;
    }
  }
};

template<RegionType region_type>
void basic_test()
{
  using RegionClass = typename RegionType_to_class<region_type>::T;
  using C = C1<region_type>;
  using D = C2<region_type>;
  using F = F1<region_type>;
  using E = F2<region_type>;

  Alloc& alloc = ThreadAlloc::get();

  auto a = new F;
  new (a) F;
  new (a) F;

  auto b = new (a) C;
  b->f1 = new (a) C;

  RegionClass::swap_root(a, b);
  if constexpr (region_type == RegionType::Trace)
  {
    RegionTrace::gc(alloc, b);
    logger::cout() << "GCed" << std::endl;
  }

  new (b) F;
  new (b) F;
  a = new (b) F;

  RegionClass::swap_root(b, a);

  Region::release(alloc, a);

  auto c = new D;
  c->f1 = new (c) D;
  c->f1->f1 = new D; // This is a fresh region.
  Region::release(alloc, c);

  auto d = new D;
  d->f1 = new D; // This is a fresh region, hanging off the entry object.
  new (d) D;
  Region::release(alloc, d);

  auto e0 = new E(0, nullptr);
  auto e1 = new (e0) E(1, e0);
  auto e2 = new (e0) E(2, e1);
  auto e3 = new (e0) E(3, e2);
  e0->child = e1;
  e1->child = e2;
  e2->child = e3;
  Region::release(alloc, e0);

  snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
}

int main(int, char**)
{
  basic_test<RegionType::Trace>();
  basic_test<RegionType::Arena>();
  return live_count;
}
