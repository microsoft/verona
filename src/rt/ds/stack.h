// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <cassert>
#include <snmalloc.h>

namespace verona::rt
{
  /**
   * This class contains the core functionality for a stack using aligned blocks
   * of memory. The stack is the size of a single pointer when empty.
   */
  template<class T, class Alloc>
  class StackThin
  {
  private:
    static constexpr size_t POINTER_COUNT = 64;
    static_assert(
      snmalloc::bits::next_pow2_const(POINTER_COUNT) == POINTER_COUNT,
      "Should be power of 2 for alignment.");

    static constexpr size_t STACK_COUNT = POINTER_COUNT - 1;

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
    struct alignas(POINTER_COUNT * sizeof(T*)) Block
    {
      T** prev;
      T* data[STACK_COUNT];
    };

  private:
    static_assert(
      sizeof(Block) == alignof(Block), "Size and align must be equal");

    // Dummy block to effectively allow pointer arithmetic on nullptr
    // which is undefined behaviour.  So we statically allocate a block
    // to represent the end of the stack.
    inline static Block null_block{};

    // Index of the full dummy block
    // Due to pointer arithmetic with nullptr being undefined behaviour
    // we use a statically allocated null block.
    static constexpr T** null_index = &(null_block.data[STACK_COUNT - 1]);

    /// Mask to access the index component of the pointer to a block.
    static constexpr uintptr_t INDEX_MASK = (POINTER_COUNT - 1) * sizeof(T*);

    /// Pointer into a block.  As the blocks are strongly aligned
    /// the bits 9-3 represent the element in the block, with 0 being
    /// a pointer to the `prev` pointer, and implying the empty block.
    T** index;

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
    StackThin() : index(null_index)
    {
      static_assert(
        sizeof(*this) == sizeof(void*),
        "Stack should contain only the index pointer");
    }

    /// Deallocate the linked blocks for this stack.
    void dealloc(Alloc& alloc)
    {
      auto local_block = get_block(index);
      while (local_block != &null_block)
      {
        auto prev = get_block(local_block->prev);
        alloc.template dealloc<sizeof(Block)>(local_block);
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

    /// Call this to pop an element from the stack.
    ALWAYSINLINE T* pop(Alloc& alloc)
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

    /// Call this to push an element onto the stack.
    ALWAYSINLINE void push(T* item, Alloc& alloc)
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
    void forall(snmalloc::function_ref<void(T*)> apply)
    {
      T** curr = index;

      while (curr != null_index)
      {
        do
        {
          apply(*curr);
          curr--;
        } while (!is_empty(curr));

        curr = get_block(curr)->prev;
      }
    }

  private:
    /// Slow path for push, performs a push, when allocation is required.
    void push_slow(T* item, Alloc& alloc)
    {
      assert(is_full(index));

      Block* next = (Block*)alloc.template alloc<sizeof(Block)>();
      assert(((uintptr_t)next) % alignof(Block) == 0);
      next->prev = index;
      index = &(next->data[0]);
      *index = item;
    }

    /// Slow path for pop, performs a pop, when deallocation of a block is
    /// required.
    T* pop_slow(Alloc& alloc)
    {
      assert(is_empty(index - 1));

      auto item = peek();
      Block* block = get_block(index);
      index = block->prev;

      alloc.template dealloc<sizeof(Block)>(block);
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
    /**
     * The BackupAlloc allocates Blocks for the stack
     * Uses a one place pool to avoid always calling the allocator.
     */
    class BackupAlloc
    {
      using Block = typename StackThin<T, BackupAlloc>::Block;

      /// A one place pool of Block.
      Block* backup = nullptr;

      /// Allocator that blocks are supplied by.
      Alloc& underlying_alloc;

    public:
      BackupAlloc(Alloc& a) : underlying_alloc(a) {}

      /// Allocate a stack Block.
      template<size_t Size>
      ALWAYSINLINE void* alloc()
      {
        static_assert(
          Size == sizeof(Block),
          "Allocating something not the size of a block");

        if (backup)
          return std::exchange(backup, nullptr);
        else
          return underlying_alloc.template alloc<Size>();
      }

      /// Deallocate a stack Block.
      template<size_t Size>
      ALWAYSINLINE void dealloc(Block* b)
      {
        static_assert(
          Size == sizeof(Block),
          "Deallocating something not the size of a block");

        if (backup == nullptr)
          backup = b;
        else
          underlying_alloc.template dealloc<Size>(b);
      }

      ~BackupAlloc()
      {
        if (backup != nullptr)
          underlying_alloc.template dealloc<sizeof(Block)>(backup);
      }
    };

    /// Underlying stack
    StackThin<T, BackupAlloc> stack;

    /// Allocator for new blocks of stack
    BackupAlloc backup_alloc;

  public:
    Stack(Alloc& alloc) : backup_alloc(alloc) {}

    /// Return top element of the stack
    ALWAYSINLINE T* peek()
    {
      return stack.peek();
    }

    /// Put an element on the stack
    ALWAYSINLINE void push(T* item)
    {
      stack.push(item, backup_alloc);
    }

    /// Remove an element on the stack
    ALWAYSINLINE T* pop()
    {
      return stack.pop(backup_alloc);
    }

    /// Check if stack is empty
    ALWAYSINLINE bool empty()
    {
      return stack.empty();
    }

    /// Apply function to every element of the stack.
    ALWAYSINLINE
    void forall(snmalloc::function_ref<void(T*)> apply)
    {
      stack.forall(apply);
    }

    ~Stack()
    {
      stack.dealloc(backup_alloc);
    }
  };
} // namespace verona::rt
