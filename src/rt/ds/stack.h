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
  class StackSmall
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

    /// Checks if an index into a block means the block has no space.
    static bool is_full(T** index)
    {
      return ((uintptr_t)index & INDEX_MASK) == INDEX_MASK;
    }

  public:
    StackSmall() : index(null_index)
    {
      static_assert(
        sizeof(*this) == sizeof(void*),
        "Stack should contain only the index pointer");
    }

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

    /// Call this to pop an element from the stack.  Only
    /// correct to call this if pop_is_fast just returned
    /// true.
    ALWAYSINLINE T* pop(Alloc* alloc)
    {
      assert(!empty());
      if (!is_empty(index - 1))
      {
        auto item = peek();
        index--;
        return item;
      }

      return pop_slow(alloc);
    }

    /// Call this to push an element onto the stack.  Only
    /// correct to call this if push_is_fast just returned
    /// true.
    ALWAYSINLINE void push(T* item, Alloc* alloc)
    {
      if (!is_full(index))
      {
        index++;
        *index = item;
        return;
      }

      push_slow(item, alloc);
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

  private:
    /// Call this to push an element onto the stack.  Only
    /// correct to call this if push_is_fast just returned
    /// false.  It needs to be provided a new block of memory
    /// for the stack to use.
    void push_slow(T* item, Alloc* alloc)
    {
      assert(is_full(index));

      Block* block = (Block*)alloc->template alloc<sizeof(Block)>();
      T** iter = (T**)block;
      assert(is_empty(iter));
      auto next = get_block(iter);

      assert(index != (T**)&null_block);
      next->prev = index;
      index = &(next->data[0]);
      next->data[0] = item;
    }

    /// Call this to pop an element from the stack.  Only
    /// correct to call this if pop_is_fast just returned
    /// false.  This returns a pair of the popped element
    /// and the block that the client must dispose of.
    T* pop_slow(Alloc* alloc)
    {
      assert(is_empty(index - 1));

      auto item = peek();
      Block* block = get_block(index);
      index = block->prev;

      alloc->template dealloc<sizeof(Block)>(block);
      return item;
    }
  };

  /**
   * This class uses the block structured stack with extra fields
   * for the allocator, and a backup block, so that the common case
   * of 0-1 elements can be fast, and any other block boundrary case.
   *
   * As this data-structure keeps a copy of the allocator, it must not
   * be passed between threads.  Use `StackSmall` for that kind of use case.
   * When pushing, this class will use the backup block rather than allocate and
   * when popping will return a block to the backup pointer rather than
   * deallocate. This means that we experience allocate / deallocate churn only
   * when rapidly crossing two allocation boundaries. A loop that pushes 32
   * elements and pops them on each iteration may trigger allocation the first
   * time but will then not trigger allocation on any subsequent iteration.
   */
  template<class T, class Alloc>
  class Stack
  {
    class BackupAlloc
    {
      using Block = typename StackSmall<T, BackupAlloc>::Block;
      Block* backup = nullptr;
      Alloc* underlying_alloc;

    public:
      BackupAlloc(Alloc* a) : underlying_alloc(a) {}

      template<size_t Size>
      ALWAYSINLINE void* alloc()
      {
        static_assert(
          Size == sizeof(Block),
          "Allocating something not the size of a block");

        if (backup)
          return std::exchange(backup, nullptr);
        else
          return underlying_alloc->template alloc<Size>();
      }

      template<size_t Size>
      ALWAYSINLINE void dealloc(Block* b)
      {
        static_assert(
          Size == sizeof(Block),
          "Deallocating something not the size of a block");

        if (backup == nullptr)
          backup = b;
        else
          underlying_alloc->template dealloc<Size>(b);
      }

      ~BackupAlloc()
      {
        if (backup != nullptr)
          underlying_alloc->template dealloc<sizeof(Block)>(backup);
      }
    };

    StackSmall<T, BackupAlloc> stack;
    BackupAlloc backup_alloc;

  public:
    Stack(Alloc* alloc) : backup_alloc(alloc) {}

    ALWAYSINLINE T* peek()
    {
      return stack.peek();
    }

    ALWAYSINLINE void push(T* item)
    {
      stack.push(item, &backup_alloc);
    }

    ALWAYSINLINE T* pop()
    {
      return stack.pop(&backup_alloc);
    }

    ALWAYSINLINE bool empty()
    {
      return stack.empty();
    }

    ~Stack()
    {
      stack.dealloc(&backup_alloc);
    }
  };
} // namespace verona::rt
