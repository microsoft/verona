// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <cassert>
#include <snmalloc.h>

namespace verona::rt
{
  /**
   * This class contains the core functionality for a stack using aligned blocks
   * of memory. It is not expecting to be used directly, but for one of its
   * wrappers below to be used which correctly handle allocation.
   */
  template<class T, class Alloc>
  class StackBase
  {
  private:
    static constexpr size_t STACK_COUNT = 63;

    /**
     * The assumes that the allocations are aligned to the same threshold as
     * their size. The blocks contain one previous pointer, and 63 pointers to
     * Ts.  This is a power of two, so we can use the bottom part of the
     * pointer to track the index.
     *
     * As the block contains a previous pointer, there are only 64 possible
     * states for a block, that is 0 - 63 live entries.
     *
     * The stack is represented by a single interior pointer, index, of type
     * T**.
     *
     * Note that `index` can point to a `prev` element of a block,
     * and thus be mistyped. This represents the empty block and is never
     * followed directly.
     */
  public:
    struct alignas((STACK_COUNT + 1) * sizeof(T*)) Block
    {
      T** prev;
      T* data[STACK_COUNT];
    };

    // Dummy block to effectively allow pointer arithmetic on nullptr
    // which is undefined behaviour.  So we statically allocate a block
    // to represent the end of the stack.
    inline static Block null_block{};

    // Index of the full dummy block
    // Due to pointer arithmetic with nullptr being undefined behaviour
    // we use a statically allocated null block.
    static constexpr T** null_index = &(null_block.data[STACK_COUNT - 1]);

  private:
    /// Mask to access the index component of the pointer to a block.
    static constexpr uintptr_t INDEX_MASK = STACK_COUNT * sizeof(T*);

    /// Pointer into a block.  As the blocks are strongly aligned
    /// the bits 9-3 represent the element in the block, with 0 being
    /// a pointer to the `prev` pointer, and implying the empty block.
    T** index;

  private:
    /// Takes an index and returns the pointer to the Block
    static Block* get_block(T** ptr)
    {
      return snmalloc::pointer_align_down<sizeof(Block), Block>(ptr);
    }

    /// Checks if an index into a block means the block is empty.
    static bool is_empty(T** ptr)
    {
      return ((uintptr_t)ptr & INDEX_MASK) == 0;
    }

    /// Checks if an index into a block means the block has space.
    static bool is_not_full(T** index)
    {
      return ((uintptr_t)index & INDEX_MASK) != INDEX_MASK;
    }

  public:
    StackBase() : index(null_index) {}

    /// Deallocate the linked blocks for this stack.
    void dealloc(Alloc* alloc)
    {
      auto local_block = get_block(index);
      while (local_block != &null_block)
      {
        auto prev = get_block(local_block->prev);
        alloc->template dealloc<sizeof(Block)>(local_block);
        local_block = prev;
      }
    }

    /// returns true if this stack is empty
    ALWAYSINLINE bool empty()
    {
      return index == null_index;
    }

    /// Return the top element of the stack without removing it.
    ALWAYSINLINE T* peek()
    {
      assert(!empty());
      return *index;
    }

    /// Call this to determine if pop can proceed without deallocation
    ALWAYSINLINE bool pop_is_fast()
    {
      assert(!empty());
      return !is_empty(index - 1);
    }

    /// Call this to pop an element from the stack.  Only
    /// correct to call this if pop_is_fast just returned
    /// true.
    ALWAYSINLINE T* pop_fast()
    {
      assert(pop_is_fast());
      auto item = peek();
      index--;
      return item;
    }

    /// Call this to pop an element from the stack.  Only
    /// correct to call this if pop_is_fast just returned
    /// false.  This returns a pair of the popped element
    /// and the block that the client must dispose of.
    std::pair<T*, Block*> pop_slow()
    {
      assert(!pop_is_fast());
      auto item = peek();
      T** prev_index = get_block(index)->prev;
      auto dealloc = get_block(index);
      index = prev_index;
      return {item, dealloc};
    }

    /// Call this to determine if push can proceed without
    /// a new block.
    ALWAYSINLINE bool push_is_fast()
    {
      return is_not_full(index);
    }

    /// Call this to push an element onto the stack.  Only
    /// correct to call this if push_is_fast just returned
    /// true.
    ALWAYSINLINE void push_fast(T* item)
    {
      assert(push_is_fast());
      index++;
      *index = item;
    }

    /// Call this to push an element onto the stack.  Only
    /// correct to call this if push_is_fast just returned
    /// false.  It needs to be provided a new block of memory
    /// for the stack to use.
    void push_slow(T* item, Block* block)
    {
      assert(!push_is_fast());

      T** iter = (T**)block;
      assert(is_empty(iter));
      auto next = get_block(iter);

      assert(index != (T**)&null_block);
      next->prev = index;
      index = &(next->data[0]);
      next->data[0] = item;
    }

    /// For all elements of the stack
    template<void apply(T* t)>
    void forall()
    {
      T* curr = index;

      while (curr != null_index)
      {
        do
        {
          apply(*curr);
          curr--;
        } while (is_empty(curr));

        curr = get_block(curr)->prev;
      }
    }
  };

  /**
   * This class uses the block structured stack with extra fields
   * for the allocator, and a backup block, so that the common case
   * of 0-1 elements can be fast, and any other block boundrary case.
   *
   * As this data-structure keeps a copy of the allocator, it must not
   * be passed between threads.  Use `StackSmall` for that kind of use case.
   */
  template<class T, class Alloc>
  class Stack
  {
    using Block = typename StackBase<T, Alloc>::Block;
    StackBase<T, Alloc> stack;
    Alloc* alloc;
    Block* backup;

  public:
    Stack(Alloc* alloc) : alloc(alloc), backup(nullptr) {}

    ALWAYSINLINE void push(T* item)
    {
      if (stack.push_is_fast())
      {
        stack.push_fast(item);
        return;
      }

      Block* new_block = backup;

      if (new_block == nullptr)
        new_block = (Block*)alloc->template alloc<sizeof(Block)>();

      backup = nullptr;

      stack.push_slow(item, new_block);
    }

    T* pop()
    {
      if (stack.pop_is_fast())
      {
        return stack.pop_fast();
      }

      auto res_block = stack.pop_slow();
      if (backup == nullptr)
      {
        backup = res_block.second;
      }
      else
      {
        alloc->template dealloc<sizeof(Block)>(res_block.second);
      }
      return res_block.first;
    }

    ~Stack()
    {
      stack.dealloc(alloc);
      if (backup != nullptr)
      {
        alloc->template dealloc<sizeof(Block)>(backup);
      }
    }

    bool empty()
    {
      return stack.empty();
    }

    T* peek()
    {
      return stack.peek();
    }
  };

  /**
   * Block structured stack, that is a single pointer in size.
   *
   * Operations require an explicit Alloc parameter in case more/less is
   * required.
   */
  template<class T, class Alloc>
  class StackSmall
  {
    using Block = typename StackBase<T, Alloc>::Block;

    StackBase<T, Alloc> stack;

  public:
    StackSmall() {}

    T* pop(Alloc* alloc)
    {
      if (stack.pop_is_fast())
        return stack.pop_fast();

      auto res_block = stack.pop_slow();
      alloc->template dealloc<sizeof(Block)>(res_block.second);
      return res_block.first;
    }

    void push(T* item, Alloc* alloc)
    {
      if (stack.push_is_fast())
      {
        stack.push_fast(item);
        return;
      }

      auto b = (Block*)alloc->template alloc<sizeof(Block)>();
      stack.push_slow(item, b);
    }

    void dealloc(Alloc* alloc)
    {
      stack.dealloc(alloc);
    }
  };
} // namespace verona::rt
