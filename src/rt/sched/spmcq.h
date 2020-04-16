// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "epoch.h"

namespace verona::rt
{
  /**
   * Single Produce Multiple Consumer Queue.
   *
   * This queue forms the primary scheduler queue for each thread to
   * schedule cowns.
   *
   * Slightly mis-named queue. It has two ends
   *
   *   - the first end can only be accessed from a single thread using
   *     `push` to add elements to the queue in a LIFO way wrt to `pop`.
   *   - the second end can be used by multiple threads to `pop` elements
   *     and `push_back` elements. `push_back` behaves in a FIFO way wrt to
   *     `pop`.
   *
   * The queue uses an intrusive list in the elements of the queue.  (For
   * Verona this is the Cowns). To make this memory safe and ABA safe we use
   * to mechanisms.
   *
   *   - ABA protection from snmalloc - this will use LL/SC or Double-word
   *     compare and swap.  This ensures that the same element can be added
   *     to the queue multiple times without leading to ABA issues.
   *   - Memory safety, the underlying elements of the queue my also be
   *     deallocated however, if this occurs, then we could potentially access
   *     decommitted memory with the optimistic concurrency. To protect against
   *     this we use an epoch mechanism, that is, elements may only
   *     deallocated, if sufficient epochs have passed since it was last in
   *     the queue.
   *
   * Using two mechanisms means that we can have intrusive `next` fields,
   * which gives zero allocation scheduling, but don't have to wait for the
   * epoch to advance to reschedule.
   *
   * The queue also has a notion of a token. This is used to determine once
   * the queue has been flushed through.  The client can check if the value
   * popped is a token.  This is used to monitor how quickly this queue is
   * completed, and then can be used for
   *   - The leak detector algorithm
   *   - The fairness of scheduling
   */
  template<class T>
  class SPMCQ
  {
  private:
    friend T;
    static constexpr uintptr_t BIT = 1;
    // Written by a single thread that owns the queue.
    T* head;
    // Multi-threaded end of the "queue" requires ABA protection.
    // Used for work stealing and posting new work from another thread.
    snmalloc::ABA<T> tail;

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
      UNUSED(alloc);
      auto unmasked_node = unmask(node);
      unmasked_node->next_in_queue = nullptr;
      auto unmasked_head = unmask(head);
      unmasked_head->next_in_queue.store(node, std::memory_order_release);
      head = node;
    }

    void push_back(Alloc* alloc, T* node)
    {
      UNUSED(alloc);
      auto cmp = tail.read();

      do
      {
        node->next_in_queue = cmp.ptr();
      } while (!cmp.store_conditional(node));
    }

    T* pop(Alloc* alloc)
    {
      T* next;
      T* tl;

      // Hold epoch to ensure that the value read from `tail` cannot be
      // deallocated during this operation.  This must occur before read of
      // tail.
      Epoch e(alloc);
      uint64_t epoch = e.get_local_epoch_epoch();

      auto cmp = tail.read();
      do
      {
        tl = cmp.ptr();
        auto unmasked_tl = unmask(tl);
        // This operation is memory safe due to holding the epoch.
        next = unmasked_tl->next_in_queue;

        if (next == nullptr)
          return nullptr;
      } while (!cmp.store_conditional(next));

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
