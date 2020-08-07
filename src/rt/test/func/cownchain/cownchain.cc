// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include <test/harness.h>
#include <test/xoroshiro.h>
/**
 * This example is design to test a long chain of cowns being collected.
 **/

struct ChainCown;

struct LinkObject : public V<LinkObject>
{
  ChainCown* next;

  LinkObject(ChainCown* next) : next(next) {}

  void trace(ObjectStack& fields) const;
};

struct ChainCown : public VCown<ChainCown>
{
  LinkObject* next;

  ChainCown(LinkObject* next) : next(next) {}

  static ChainCown* make_chain(Alloc* alloc, size_t length)
  {
    ChainCown* hd = nullptr;
    for (; length > 0; length--)
    {
      auto next = new (alloc) LinkObject(hd);
      if (hd != nullptr)
        RegionTrace::insert<TransferOwnership::YesTransfer>(alloc, next, hd);
      hd = new (alloc) ChainCown(next);
    }
    return hd;
  }

  void trace(ObjectStack& fields) const
  {
    if (next != nullptr)
    {
      fields.push(next);
    }
  }
};

void LinkObject::trace(ObjectStack& fields) const
{
  if (next != nullptr)
  {
    fields.push(next);
  }
}

int main(int, char**)
{
  auto alloc = ThreadAlloc::get();
  auto a = ChainCown::make_chain(alloc, 100000);
  Cown::release(alloc, a);
  snmalloc::current_alloc_pool()->debug_check_empty();
}