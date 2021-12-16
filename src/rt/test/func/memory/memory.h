// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <test/harness.h>
#include <verona.h>

using namespace snmalloc;
using namespace verona::rt;
using namespace verona::rt::api;

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

struct F1 : public V<F1>
{
  F1* f1 = nullptr;
  F1* f2 = nullptr;

  void trace(ObjectStack& st) const
  {
    if (f1 != nullptr)
      st.push(f1);

    if (f2 != nullptr)
      st.push(f2);
  }

  void finaliser(Object* region, ObjectStack& sub_regions)
  {
    Object::add_sub_region(f1, region, sub_regions);
    Object::add_sub_region(f2, region, sub_regions);
  }

  F1()
  {
    live_count++;
  }

  ~F1()
  {
    live_count--;
  }
};

template<size_t N>
struct C2 : public V<C2<N>>
{
  uint8_t data[N - sizeof(Object)];
};

template<size_t N>
struct F2 : public V<F2<N>>
{
  uint8_t data[N - sizeof(Object)];

  F2()
  {
    live_count++;
  }

  ~F2()
  {
    live_count--;
  }
};

// Foward declaration
struct F3;

struct C3 : public V<C3>
{
  C3* c1 = nullptr;
  C3* c2 = nullptr;
  F3* f1 = nullptr;
  F3* f2 = nullptr;

  void trace(ObjectStack& st) const;
};

struct F3 : public V<F3>
{
  C3* c1 = nullptr;
  C3* c2 = nullptr;
  F3* f1 = nullptr;
  F3* f2 = nullptr;

  void trace(ObjectStack& st) const
  {
    if (c1 != nullptr)
      st.push(c1);

    if (c2 != nullptr)
      st.push(c2);

    if (f1 != nullptr)
      st.push(f1);

    if (f2 != nullptr)
      st.push(f2);
  }

  void finaliser(Object* region, ObjectStack& sub_regions)
  {
    Object::add_sub_region(c1, region, sub_regions);
    Object::add_sub_region(c2, region, sub_regions);
    Object::add_sub_region(f1, region, sub_regions);
    Object::add_sub_region(f2, region, sub_regions);
  }

  F3()
  {
    live_count++;
  }

  ~F3()
  {
    live_count--;
  }
};

void C3::trace(ObjectStack& st) const
{
  if (c1 != nullptr)
    st.push(c1);

  if (c2 != nullptr)
    st.push(c2);

  if (f1 != nullptr)
    st.push(f1);

  if (f2 != nullptr)
    st.push(f2);
}

// Only two can fit into an Arena.
using MediumC2 = C2<400 * 1024 - 4 * sizeof(uintptr_t)>;
using MediumF2 = F2<400 * 1024 - 4 * sizeof(uintptr_t)>;

// Fits exactly into an Arena.
using LargeC2 = C2<1024 * 1024 - 4 * sizeof(uintptr_t)>;
using LargeF2 = F2<1024 * 1024 - 4 * sizeof(uintptr_t)>;

// Too large for Arena.
using XLargeC2 = C2<1024 * 1024 - 4 * sizeof(uintptr_t) + 1>;
using XLargeF2 = F2<1024 * 1024 - 4 * sizeof(uintptr_t) + 1>;

/**
 * Allocates objects of types First and Rest... into a region represented by
 * iso object o.
 **/
template<size_t n, class First, class... Rest>
auto allocs()
{
  auto a = new First;
  if constexpr (sizeof...(Rest) > 0)
  {
    auto b = allocs<n - 1, Rest...>();
    if constexpr (n == 0)
    {
      UNUSED(b);
      return a;
    }
    else
    {
      UNUSED(a);
      return b;
    }
  }
  else
  {
    return a;
  }
}

template<size_t n, class First, class... Rest>
auto alloc_in_region(Object* r)
{
  UsingRegion rr(r);
  return allocs<n, First, Rest...>();
}

/**
 * Allocates a region containing objects of types First and Rest..., where
 * First is the type of the iso object. Returns a pointer to the iso object.
 *
 * This helper is used by many different tests, so we don't release the region.
 **/
template<class First, class... Rest>
First* alloc_region(RegionType rt)
{
  auto o = new (rt) First;
  {
    if constexpr (sizeof...(Rest) > 0)
      alloc_in_region<0, Rest...>(o);
  }
  return o;
}
