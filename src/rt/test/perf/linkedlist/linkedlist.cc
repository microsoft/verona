// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <iomanip>
#include <iostream>
#include <test/measuretime.h>
#include <verona.h>

using namespace snmalloc;
using namespace verona::rt;
using namespace verona::rt::api;

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

void test_linked_list()
{
  // Freeze a doubly linked list

  auto& alloc = ThreadAlloc::get();

  for (int list_size = 100000; list_size <= 1000000; list_size += 100000)
  {
    C1* curr;
    C1* root;
    C1* next;

    {
      MeasureTime m;
      m << "Alloc DLL:  " << std::setw(10) << list_size;
      curr = new (RegionType::Trace) C1;
      root = curr;
      curr->f2 = nullptr;
      next = nullptr;

      {
        UsingRegion rr(root);
        for (int i = 0; i < list_size; i++)
        {
          next = new C1;
          curr->f1 = next;
          next->f2 = curr;
          curr = next;
        }
      }
      curr->f1 = nullptr;
    }

    {
      MeasureTime m;
      m << "Freeze DLL: " << std::setw(10) << list_size;
      freeze(root);
    }

    // Free immutable graph.
    {
      MeasureTime m;
      m << "Free DLL:   " << std::setw(10) << list_size;
      Immutable::release(alloc, root);
    }
  }

  snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
}

int main(int, char**)
{
  test_linked_list();
  return 0;
}
