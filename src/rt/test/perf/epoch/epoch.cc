// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <test/measuretime.h>
#include <test/opt.h>
#include <verona.h>

using namespace snmalloc;
using namespace verona::rt;

void test_epoch()
{
  // Used to prevent malloc from being optimised away.
  static void* old = nullptr;

  auto& alloc = ThreadAlloc::get();
  constexpr int count = 10000000;
  constexpr int size = 48;
  void* special = alloc.alloc(size);
  void* obj = nullptr;

  {
    MeasureTime m;
    m << "with_epoch   ";
    for (int n = 0; n < count; n++)
    {
      Epoch e(alloc);
      obj = alloc.alloc(size);
      e.delete_in_epoch(obj);
    }

    Epoch::flush(alloc);
  }

  {
    MeasureTime m;
    m << "without_epoch";
    for (int n = 0; n < count; n++)
    {
      obj = alloc.alloc(size);
      old = obj;
      alloc.dealloc(obj, size);
    }
  }

  {
    MeasureTime m;
    m << "template_no_e";
    for (int n = 0; n < count; n++)
    {
      obj = alloc.alloc<size>();
      old = obj;
      alloc.dealloc<size>(obj);
    }
  }

  alloc.dealloc(special);
  snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
  (void)old;
}

int main(int, char**)
{
  test_epoch();
  return 0;
}
