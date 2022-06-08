// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <test/harness.h>
#include <test/log.h>
#include <verona.h>

using namespace snmalloc;
using namespace verona::rt;

static std::atomic<int> live_count;

struct C1 : public V<C1>
{
  C1* f1 = nullptr;
  C1* f2 = nullptr;

  void trace(ObjectStack& st) const
  {
    if (f1 != nullptr)
      st.push(f1);

    if (f2 != nullptr)
      st.push(f2);
  }
};

class C2 : public V<C2>
{
public:
  C2* f1 = nullptr;
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

class F1 : public V<F1>
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

class F2 : public V<F2>
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
  using C = C1;
  using D = C2;
  using F = F1;
  using E = F2;
  auto a = new (region_type) F;
  {
    UsingRegion ur(a);
    new F;
    new F;

    auto b = new C;
    b->f1 = new C;

    set_entry_point(b);
    if constexpr (region_type == RegionType::Trace)
    {
      region_collect();
      logger::cout() << "GCed" << std::endl;
    }

    new F;
    new F;
    a = new F;

    set_entry_point(a);
  }
  region_release(a);

  auto c = new (region_type) D;
  {
    UsingRegion ur(c);
    c->f1 = new D;
    c->f1->f1 = new (region_type) D; // This is a fresh region.
  }
  region_release(c);

  auto d = new (region_type) D;
  {
    UsingRegion ur(d);
    d->f1 = new (region_type)
      D; // This is a fresh region, hanging off the entry object.
    new D;
  }
  region_release(d);

  auto e0 = new (region_type) E(0, nullptr);
  {
    UsingRegion ur(e0);
    auto e1 = new E(1, e0);
    auto e2 = new E(2, e1);
    auto e3 = new E(3, e2);
    e0->child = e1;
    e1->child = e2;
    e2->child = e3;
  }
  region_release(e0);

  snmalloc::debug_check_empty<snmalloc::Alloc::Config>();
}

int main(int, char**)
{
  basic_test<RegionType::Trace>();
  basic_test<RegionType::Arena>();
  return live_count;
}
