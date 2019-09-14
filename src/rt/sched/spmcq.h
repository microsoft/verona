// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "epoch.h"

namespace verona::rt
{
  template<class T>
  class SPMCQ
  {
  private:
    friend T;
    T* head;
    snmalloc::ABA<T> tail;
    static constexpr uintptr_t BIT = 1;

    T* unmask(T* tagged_ptr)
    {
      return (T*)((uintptr_t)tagged_ptr & ~BIT);
    }

    bool is_bit_set(T* tagged_ptr)
    {
      return unmask(tagged_ptr) != tagged_ptr;
    }

    T* set_bit(T* ptr)
    {
      return (T*)((uintptr_t)ptr | BIT);
    }

  public:
    explicit SPMCQ(T* token)
    {
      assert(token);
      token->next_in_queue = nullptr;
      token = set_bit(token);
      head = token;
      tail.init(token);
    }

    void push(Alloc* alloc, T* node)
    {
      (void)alloc;
      auto unmasked_node = unmask(node);
      unmasked_node->next_in_queue = nullptr;
      auto unmasked_head = unmask(head);
      unmasked_head->next_in_queue.store(node, std::memory_order_release);
      head = node;
    }

    void push_back(Alloc* alloc, T* node)
    {
      (void)alloc;
      auto cmp = tail.read();

      do
      {
        node->next_in_queue = ABA<T>::ptr(cmp);
      } while (!tail.compare_exchange(cmp, node));
    }

    T* pop(Alloc* alloc)
    {
      T* next;
      T* tl;
      auto cmp = tail.read();

      uint64_t epoch;
      do
      {
        Epoch e(alloc);
        epoch = e.get_local_epoch_epoch();
        tl = ABA<T>::ptr(cmp);
        auto unmasked_tl = unmask(tl);
        next = unmasked_tl->next_in_queue;

        if (next == nullptr)
          return nullptr;
      } while (!tail.compare_exchange(cmp, next));

      assert(epoch != T::NO_EPOCH_SET);

      tl->epoch_when_popped = epoch;

      return tl;
    }

    // The callers are expected to guarantee no one is attempting to access the
    // queue concurrently.
    void destroy(Alloc* alloc)
    {
      assert(tail.peek() == head);
      assert(is_bit_set(head));
      auto unmasked = unmask(head);
      assert(unmasked->next_in_queue == nullptr);

      unmasked->dealloc(alloc);
    }

    bool is_empty()
    {
      return head == tail.peek() && is_bit_set(head);
    }
  };
} // namespace verona::rt
