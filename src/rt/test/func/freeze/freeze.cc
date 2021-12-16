// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "test/harness.h"
#include "test/xoroshiro.h"

#include <vector>
#include <verona.h>

using namespace snmalloc;
using namespace verona::rt;
using namespace verona::rt::api;

// This only tests trace regions.
// At the moment, it does not make sense to freeze arena regions.

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

  void finaliser(Object* region, ObjectStack& st)
  {
    if (f1 != nullptr)
    {
      Object::add_sub_region(f1, region, st);
    }
    if (f2 != nullptr)
    {
      Object::add_sub_region(f2, region, st);
    }
  }
};

class Foo : public V<Foo>
{
public:
  int value = -1;
};

template<class T>
class List : public V<List<T>>
{
private:
  T* head = nullptr;
  List<T>* tail = nullptr;

public:
  void init(T* elem, List<T>* tl)
  {
    head = elem;
    tail = tl;
  }

  T* element()
  {
    return head;
  }

  List<Foo>* next()
  {
    return tail;
  }

  // Required by the library;
  void trace(ObjectStack& st) const
  {
    if (head != nullptr)
      st.push(head);

    if (tail != nullptr)
      st.push(tail);
  }
};

void test1()
{
  // Freeze an scc.
  // 1 -> 2
  // 2 -> 1
  auto& alloc = ThreadAlloc::get();

  C1* r = new (RegionType::Trace) C1;
  {
    UsingRegion r2(r);
    r->f1 = new C1;
    r->f1->f1 = r;
  }

  freeze(r);

  auto rr = r->debug_immutable_root();
  UNUSED(rr);
  check(rr->debug_test_rc(1));
  check(r->f1->debug_immutable_root() == rr);
  check(r->f1->debug_test_rc(1));

  // Free immutable graph.
  Immutable::release(alloc, r);

  snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
}

void test2()
{
  // Freeze a more complex scc.
  // 1 -> 2, 3 = scc rc 1
  // 2 -> 3, 4 = scc rc 1
  // 3         = scc rc 4
  // 4 -> 3, 5 = ptr 2
  // 5 -> 2, 6 = ptr 2
  // 6 -> 4, 3 = ptr 2
  auto& alloc = ThreadAlloc::get();

  C1* o1 = new (RegionType::Trace) C1;
  C1 *o2, *o3, *o4, *o5, *o6;
  {
    UsingRegion r(o1);

    o2 = new C1;
    o3 = new C1;
    o4 = new C1;
    o5 = new C1;
    o6 = new C1;

    o1->f1 = o2;
    o1->f2 = o3;
    o2->f1 = o3;
    o2->f2 = o4;
    o4->f1 = o3;
    o4->f2 = o5;
    o5->f1 = o2;
    o5->f2 = o6;
    o6->f1 = o4;
    o6->f2 = o3;
  }

  freeze(o1);

  check(o1->debug_immutable_root() == o1);
  check(o1->debug_test_rc(1));

  auto r2 = o2->debug_immutable_root();
  UNUSED(r2);
  check(r2->debug_test_rc(1));
  check(o4->debug_immutable_root() == r2);
  check(o5->debug_immutable_root() == r2);
  check(o6->debug_immutable_root() == r2);

  check(o3->debug_immutable_root() == o3);
  check(o3->debug_test_rc(4));

  // Free immutable graph.
  Immutable::release(alloc, o1);

  snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
}

void test3()
{
  // Freeze a more complex scc.
  // 1 -> 2, 2 = scc rc 1
  // 2 -> 3, 9 = scc rc 1
  // 3 -> 4, 7 = scc rc 1
  // 4 -> 5, 6 = scc rc 1
  // 5 -> 3, 3 = scc rc 1
  // 6 -> 2, 2 = scc rc 1
  // 7 -> 8, 8 = scc rc 1
  // 8 -> 4, 4 = scc rc 1
  // 9 -> 1, 1 = scc rc 1
  auto& alloc = ThreadAlloc::get();

  auto o1 = new (RegionType::Trace) C1;
  C1 *o2, *o3, *o4, *o5, *o6, *o7, *o8, *o9;
  {
    UsingRegion r(o1);
    o2 = new C1;
    o3 = new C1;
    o4 = new C1;
    o5 = new C1;
    o6 = new C1;
    o7 = new C1;
    o8 = new C1;
    o9 = new C1;

    o1->f1 = o2;
    o1->f2 = o2;
    o2->f1 = o3;
    o2->f2 = o9;
    o3->f1 = o4;
    o3->f2 = o7;
    o4->f1 = o5;
    o4->f2 = o6;
    o5->f1 = o3;
    o5->f2 = o3;
    o6->f1 = o2;
    o6->f2 = o2;
    o7->f1 = o8;
    o7->f2 = o8;
    o8->f1 = o4;
    o8->f2 = o4;
    o9->f1 = o1;
    o9->f2 = o1;
  }

  freeze(o1);

  auto r = o1->debug_immutable_root();
  UNUSED(r);
  check(o1->debug_test_rc(1));
  check(o2->debug_immutable_root() == r);
  check(o3->debug_immutable_root() == r);
  check(o4->debug_immutable_root() == r);
  check(o5->debug_immutable_root() == r);
  check(o6->debug_immutable_root() == r);
  check(o7->debug_immutable_root() == r);
  check(o8->debug_immutable_root() == r);
  check(o9->debug_immutable_root() == r);

  // Free immutable graph.
  Immutable::release(alloc, o1);

  snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
}

void test4()
{
  // Freeze multiple regions at once
  //
  // There are three regions, [1], [2,3,4] and [5].
  //
  // 1 -> 2    = scc rc 1
  // 2 -> 3    = scc rc 1
  // 3 -> 4    = ptr 2
  // 4 -> 2, 5 = ptr 2
  // 5         = scc rc 1
  auto& alloc = ThreadAlloc::get();

  C1* o1 = new (RegionType::Trace) C1;
  C1* o2 = new (RegionType::Trace) C1;
  C1 *o3, *o4, *o5;
  {
    UsingRegion r(o2);
    o3 = new C1;
    o4 = new C1;
    o2->f1 = o3;
    o3->f1 = o4;
    o4->f1 = o2;

    o5 = new (RegionType::Trace) C1;
    o4->f2 = o5;
  }

  {
    UsingRegion r(o1);
    o1->f1 = o2;
  }

  freeze(o1);

  check(o1->debug_test_rc(1));

  auto r = o2->debug_immutable_root();
  UNUSED(r);
  check(o2->debug_test_rc(1));
  check(o3->debug_immutable_root() == r);
  check(o4->debug_immutable_root() == r);

  check(o5->debug_test_rc(1));

  // Free immutable graph.
  Immutable::release(alloc, o1);

  snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
}

void test5()
{
  // Freeze with unreachable subregion
  // Bug reported in #83
  //
  // There are two regions, [1, 2], [3].
  //
  // Freeze 1,
  // Ptr from 2 to subregion 3
  auto& alloc = ThreadAlloc::get();

  C1* o1 = new (RegionType::Trace) C1;
  std::cout << "o1: " << o1 << std::endl;
  {
    UsingRegion r(o1);
    C1* o2 = new C1;
    std::cout << "o2: " << o2 << std::endl;
    C1* o3 = new (RegionType::Trace) C1;
    std::cout << "o3: " << o3 << std::endl;
    o2->f1 = o3;
  }

  freeze(o1);

  check(o1->debug_test_rc(1));

  // Free immutable graph.
  Immutable::release(alloc, o1);

  snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
}

void freeze_weird_ring()
{
  auto& alloc = ThreadAlloc::get();

  auto root = new (RegionType::Trace) List<Foo>;

  {
    UsingRegion r(root);
    List<Foo>* next = new List<Foo>;
    List<Foo>* next_next = new List<Foo>;

    Foo* foo1 = new Foo;
    Foo* foo2 = new Foo;
    Foo* foo3 = new Foo;

    root->init(foo1, next);
    next->init(foo2, next_next);
    next_next->init(foo3, root);
  }

  freeze(root);

  // Free immutable graph.
  Immutable::release(alloc, root);

  snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
}

struct Symbolic : public V<Symbolic>
{
  size_t id;
  std::vector<Symbolic*> fields;

  void trace(ObjectStack& s) const
  {
    for (auto o : fields)
    {
      s.push(o);
    }
  }
};

void test_two_rings_1()
{
  auto& alloc = ThreadAlloc::get();

  auto r = new (RegionType::Trace) C1;
  {
    UsingRegion rr(r);
    new Symbolic;
  }
  freeze(r);
  Immutable::release(alloc, r);
  snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
}

void test_two_rings_2()
{
  auto& alloc = ThreadAlloc::get();
  auto r = new (RegionType::Trace) Symbolic;
  {
    UsingRegion rr(r);
    new C1;
  }
  freeze(r);
  Immutable::release(alloc, r);
  snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
}

void test_random(size_t seed = 1, size_t max_edges = 128)
{
  snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
  auto& alloc = ThreadAlloc::get();
  size_t id = 0;

  xoroshiro::p128r32 r(seed);
  Symbolic* root = new (RegionType::Trace) Symbolic;
#ifndef NDEBUG
  bool* reach = new bool[max_edges * max_edges];
  for (size_t i = 0; i < max_edges * max_edges; i++)
    reach[i] = false;
#endif
  std::vector<Symbolic*> all_objects;

  {
    UsingRegion rr(root);
    root->id = id++;
    all_objects.push_back(root);
    check(all_objects[root->id] == root);
#ifndef NDEBUG
    reach[root->id * max_edges + root->id] = true;
#endif
    check(root->debug_is_iso());

    size_t count = r.next() % max_edges;
    for (size_t i = 0; i < count; i++)
    {
      Symbolic* dst;
      if (r.next() % 4 == 0)
        dst = all_objects[r.next() % all_objects.size()];
      else
      {
        dst = new Symbolic;
        dst->id = id++;
        all_objects.push_back(dst);
        check(all_objects[root->id] == root);
      }
      auto src = all_objects[r.next() % all_objects.size()];
      src->fields.push_back(dst);
#ifndef NDEBUG
      reach[src->id * max_edges + dst->id] = true;
#endif
    }
  }
  freeze(root);

#ifndef NDEBUG
  // build transitive closure in rev_fields
  // This is using Floyd-Warshall's algorithm restricted to
  // reachability.
  // https://en.wikipedia.org/wiki/Floyd–Warshall_algorithm
  // This is O(N^3) the algorithm we are testing is O(N),
  // so don't do this on big graphs.
  for (size_t k = 0; k < all_objects.size(); k++)
  {
    for (size_t j = 0; j < all_objects.size(); j++)
    {
      if (j == k)
        continue;
      for (size_t i = 0; i < all_objects.size(); i++)
      {
        if (i == j || i == k)
          continue;

        if (reach[i * max_edges + k] && reach[k * max_edges + j])
          reach[i * max_edges + j] = true;
      }
    }
  }

  // Check SCCs are correct
  // If a can reach b, and b can reach a, then they should be
  // in the same scc, i.e. have the same root in the SCC algorithm.
  for (size_t i = 0; i < all_objects.size(); i++)
  {
    // Will have been collected if not reachable
    if (!reach[root->id * max_edges + i])
      continue;

    auto io = all_objects[i];
    auto ior = (Symbolic*)io->debug_immutable_root();

    for (size_t j = 0; j < all_objects.size(); j++)
    {
      // Will have been collected if not reachable
      if (!reach[root->id * max_edges + j])
        continue;
      if (i == j)
        continue;

      bool i_reaches_j = reach[i * max_edges + j];
      bool j_reaches_i = reach[j * max_edges + i];

      auto jo = all_objects[j];
      auto jor = jo->debug_immutable_root();
      bool different_root = jor != ior;
      bool in_scc = i_reaches_j && j_reaches_i;

      if (different_root == in_scc)
        abort();
    }
  }
  // Root has RC of one
  size_t* refcount = new size_t[max_edges];
  for (size_t i = 0; i < all_objects.size(); i++)
  {
    refcount[i] = i == 0 ? 1 : 0;
  }
  // For each object, and each field add an RC
  for (size_t i = 0; i < all_objects.size(); i++)
  {
    if (!reach[root->id * max_edges + i])
      continue;
    auto io = all_objects[i];
    auto ior = io->debug_immutable_root();
    for (auto ko : io->fields)
    {
      auto kor = (Symbolic*)ko->debug_immutable_root();
      // Ignore edges between the same SCC
      if (kor != ior)
        continue;
      refcount[kor->id]++;
    }
  }
  // Check all the RCs are correct
  for (size_t i = 0; i < all_objects.size(); i++)
  {
    if (!reach[root->id * max_edges + i])
      continue;
    auto io = all_objects[i];
    auto ior = (Symbolic*)io->debug_immutable_root();
    ior->debug_test_rc(refcount[ior->id]);
  }

  delete[] refcount;
  delete[] reach;
#endif

  Immutable::release(alloc, root);

  snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
}

int main(int, char**)
{
  test1();
  test2();
  test3();
  test4();
  test5();
  test_two_rings_1();
  test_two_rings_2();
  freeze_weird_ring();

  for (size_t i = 1; i < 10000; i++)
  {
    if (i % 20 == 0)
      std::cout << std::endl << i;
    std::cout << ".";
#ifndef NDEBUG
    test_random(i, 42);
#else
    test_random(i, 2400);
#endif
  }
  return 0;
}
