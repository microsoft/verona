// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "../object/object.h"

namespace verona::rt
{
  /**
   * Helper class for using the region meta data field of an object to build a
   * stack
   *
   * Used during deallocation to track sets of things to deallocate without
   * performing any allocations.
   **/
  class LinkedObjectStack
  {
  private:
    Object* header = nullptr;

  public:
    bool empty()
    {
      return header == nullptr;
    }

    Object* pop()
    {
      assert(!empty());

      auto result = header;
      header = header->get_next_any_mark();
      return result;
    }

    void push(Object* value)
    {
      value->init_next(header);
      header = value;
    }

    template<void apply(Object* t)>
    void forall()
    {
      Object* curr = header;
      while (curr != nullptr)
      {
        apply(curr);
        curr = curr->get_next_any_mark();
      }
    }
  };
} // namespace verona::rt