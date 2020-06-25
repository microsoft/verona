// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
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
   * The queue has two ends.
   *
   *   - the back end can only be accessed from a single thread using
   *     `enqueue` to add elements to the queue in a FIFO way wrt to `dequeue`.
   *   - the front end can be used by multiple threads to `dequeue` elements
   *     and `enqueue_front` elements. `enqueue_front` behaves in a LIFO way wrt
   *     to `dequeue`.
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
    T* back;
    // Multi-threaded end of the "queue" requires ABA protection.
    // Used for work stealing and posting new work from another thread.
    snmalloc::ABA<T> front;

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
      back = token;
      front.init(token);
    }

    void enqueue(Alloc* alloc, T* node)
    {
      UNUSED(alloc);
      auto unmasked_node = unmask(node);
      unmasked_node->next_in_queue = nullptr;
      auto unmasked_back = unmask(back);
      unmasked_back->next_in_queue.store(node, std::memory_order_release);
      back = node;
    }

    void enqueue_front(Alloc* alloc, T* node)
    {
      UNUSED(alloc);
      auto cmp = front.read();

      do
      {
        node->next_in_queue = cmp.ptr();
      } while (!cmp.store_conditional(node));
    }

    T* dequeue(Alloc* alloc)
    {
      T* next;
      T* fnt;

      // Hold epoch to ensure that the value read from `front` cannot be
      // deallocated during this operation.  This must occur before read of
      // front.
      Epoch e(alloc);
      uint64_t epoch = e.get_local_epoch_epoch();

      auto cmp = front.read();
      do
      {
        fnt = cmp.ptr();
        auto unmasked_fnt = unmask(fnt);
        // This operation is memory safe due to holding the epoch.
        next = unmasked_fnt->next_in_queue;

        if (next == nullptr)
          return nullptr;
      } while (!cmp.store_conditional(next));

      assert(epoch != T::NO_EPOCH_SET);

      fnt->epoch_when_popped = epoch;

      return fnt;
    }

    // The callers are expected to guarantee no one is attempting to access the
    // queue concurrently.
    void destroy(Alloc* alloc)
    {
      assert(front.peek() == back);
      assert(is_bit_set(back));
      auto unmasked = unmask(back);
      assert(unmasked->next_in_queue == nullptr);

      unmasked->dealloc(alloc);
    }

    bool is_empty()
    {
      return back == front.peek() && is_bit_set(back);
    }
  };
} // namespace verona::rt
