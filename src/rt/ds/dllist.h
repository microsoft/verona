// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <cassert>
#include <type_traits>

// Inspired by the snmalloc implementation of a double linked list.
namespace verona::rt
{
  template<
    class T,
    class Terminator = std::nullptr_t,
    bool delete_on_clear = false>
  class DLList final
  {
  private:
    static_assert(
      std::is_same<decltype(T::prev), T*>::value, "T->prev must be a T*");
    static_assert(
      std::is_same<decltype(T::next), T*>::value, "T->next must be a T*");

    T* head = Terminator();
    T* tail = Terminator();

  public:
#ifndef NDEBUG
    ~DLList()
    {
      assert(is_empty());
    }
#endif

    constexpr DLList() = default;

    DLList(DLList&& o) noexcept
    {
      head = o.head;
      tail = o.tail;

      o.head = nullptr;
      o.tail = nullptr;
    }

    DLList& operator=(DLList&& o) noexcept
    {
      head = o.head;
      tail = o.tail;

      o.head = nullptr;
      o.tail = nullptr;
      return *this;
    }

    bool is_empty()
    {
      return head == Terminator();
    }

    T* get_head()
    {
      return head;
    }

    T* get_tail()
    {
      return tail;
    }

    T* pop()
    {
      T* item = head;
      if (item != Terminator())
        remove(item);

      return item;
    }

    T* pop_tail()
    {
      T* item = tail;
      if (item != Terminator())
        remove(item);
      return item;
    }

    void insert(T* item)
    {
#ifndef NDEBUG
      debug_check_not_contains(item);
#endif
      item->next = head;
      item->prev = Terminator();
      if (head != Terminator())
        head->prev = item;
      else
        tail = item;
      head = item;
#ifndef NDEBUG
      debug_check();
#endif
    }

    void insert_back(T* item)
    {
#ifndef NDEBUG
      debug_check_not_contains(item);
#endif
      item->prev = tail;
      item->next = Terminator();

      if (tail != Terminator())
        tail->next = item;
      else
        head = item;
      tail = item;
#ifndef NDEBUG
      debug_check();
#endif
    }

    void remove(T* item)
    {
#ifndef NDEBUG
      debug_check_contains(item);
#endif
      if (item->next != Terminator())
        item->next->prev = item->prev;
      else
        tail = item->prev;

      if (item->prev != Terminator())
        item->prev->next = item->next;
      else
        head = item->next;
#ifndef NDEBUG
      debug_check();
#endif
    }

    void clear()
    {
      while (head != nullptr)
      {
        auto c = head;
        remove(c);
        if (delete_on_clear)
        {
          delete c;
        }
      }
    }

    void debug_check_contains(T* item)
    {
#ifndef NDEBUG
      debug_check();
      T* curr = head;

      while (curr != item)
      {
        assert(curr != Terminator());
        curr = curr->next;
      }
#else
      UNUSED(item);
#endif
    }

    void debug_check_not_contains(T* item)
    {
#ifndef NDEBUG
      debug_check();
      T* curr = head;

      while (curr != Terminator())
      {
        assert(curr != item);
        curr = curr->next;
      }
#else
      UNUSED(item);
#endif
    }

    void debug_check()
    {
#ifndef NDEBUG
      T* item = head;
      T* prev = Terminator();

      while (item != Terminator())
      {
        assert(item->prev == prev);
        prev = item;
        item = item->next;
      }
#endif
    }
  };
} // namespace verona::rt
