// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <test/harness.h>
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
struct F1 : public V<F1<region_type>, region_type>
{
  F1<region_type>* f1 = nullptr;
  F1<region_type>* f2 = nullptr;

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

template<size_t N, RegionType region_type>
struct C2 : public V<C2<N, region_type>, region_type>
{
  uint8_t data[N - sizeof(Object)];
};

template<size_t N, RegionType region_type>
struct F2 : public V<F2<N, region_type>, region_type>
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
template<RegionType region_type>
struct F3;

template<RegionType region_type>
struct C3 : public V<C3<region_type>, region_type>
{
  C3<region_type>* c1 = nullptr;
  C3<region_type>* c2 = nullptr;
  F3<region_type>* f1 = nullptr;
  F3<region_type>* f2 = nullptr;

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
};

template<RegionType region_type>
struct F3 : public V<F3<region_type>, region_type>
{
  C3<region_type>* c1 = nullptr;
  C3<region_type>* c2 = nullptr;
  F3<region_type>* f1 = nullptr;
  F3<region_type>* f2 = nullptr;

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

// Only two can fit into an Arena.
template<RegionType region_type>
using MediumC2 = C2<400 * 1024 - 4 * sizeof(uintptr_t), region_type>;
template<RegionType region_type>
using MediumF2 = F2<400 * 1024 - 4 * sizeof(uintptr_t), region_type>;

// Fits exactly into an Arena.
template<RegionType region_type>
using LargeC2 = C2<1024 * 1024 - 4 * sizeof(uintptr_t), region_type>;
template<RegionType region_type>
using LargeF2 = F2<1024 * 1024 - 4 * sizeof(uintptr_t), region_type>;

// Too large for Arena.
template<RegionType region_type>
using XLargeC2 = C2<1024 * 1024 - 4 * sizeof(uintptr_t) + 1, region_type>;
template<RegionType region_type>
using XLargeF2 = F2<1024 * 1024 - 4 * sizeof(uintptr_t) + 1, region_type>;

/**
 * Allocates objects of types First and Rest... into a region represented by
 * iso object o.
 **/
template<class First, class... Rest>
void alloc_in_region(Alloc& alloc, Object* o)
{
  new (alloc, o) First;
  if constexpr (sizeof...(Rest) > 0)
    alloc_in_region<Rest...>(alloc, o);
}

/**
 * Allocates a region containing objects of types First and Rest..., where
 * First is the type of the iso object. Returns a pointer to the iso object.
 *
 * This helper is used by many different tests, so we don't release the region.
 **/
template<class First, class... Rest>
First* alloc_region(Alloc& alloc)
{
  First* o = new (alloc) First;
  if constexpr (sizeof...(Rest) > 0)
    alloc_in_region<Rest...>(alloc, o);
  return o;
}
