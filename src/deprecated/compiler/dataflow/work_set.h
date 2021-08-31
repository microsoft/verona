// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/ir/ir.h"

namespace verona::compiler
{
  /**
   * Unique queue of items. Inserting an element which is already present in the
   * queue has no effect.
   *
   * WorkSet assumes elements are cheap to copy (eg. indices or pointers), as
   * it stores two copies of each.
   */
  template<typename T>
  class WorkSet
  {
  public:
    /**
     * Insert a new element to the end of the work-list, only if it is not
     * already present.
     *
     * Returns true if the element was inserted.
     */
    bool insert(T value)
    {
      assert(invariant());

      bool inserted = elements_.insert(value).second;
      if (inserted)
      {
        queue_.push_back(value);
      }
      return inserted;
    }

    /**
     * The work-list must be not be empty.
     */
    T remove()
    {
      assert(invariant());
      assert(!queue_.empty());

      T value = queue_.front();
      queue_.pop_front();
      bool erased = elements_.erase(value);
      assert(erased);

      return value;
    }

    /**
     * Returns whether or not the work-list is empty.
     */
    bool empty() const
    {
      assert(invariant());
      return queue_.empty();
    }

  private:
    bool invariant() const
    {
      return queue_.size() == elements_.size();
    }

    std::deque<T> queue_;
    std::unordered_set<T> elements_;
  };
}
