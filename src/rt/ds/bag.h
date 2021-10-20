// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <cassert>
#include <snmalloc.h>

namespace verona::rt
{
  /**
   * This class contains the core functionality for a bag using aligned blocks
   * which is optimised for constant time insertion and removal.
   *
   * Removal is O(1) because a removed item will leave a hole in the bag rather
   * than shifting remaining items down like in a bag. To reduce fragmentation
   * which can occur with deallocation churn, the bag maintains a freelist which
   * is threaded through the holes left in the bag.  Insertion of new items will
   * first query the freelist to see if a hole can be reused, otherwise items
   * are bump allocated.
   *
   * To maintain an internal freelist with no additional space requirements, the
   * item `T` must be at least 1 machine word in size.
   */
  template<class T, class U, class Alloc>
  class Bag
  {
  public:
    struct Elem
    {
      T* object;
      U metadata;
    };

  private:
    static constexpr size_t ITEM_COUNT = 32;
    static_assert(
      snmalloc::bits::next_pow2_const(ITEM_COUNT) == ITEM_COUNT,
      "Should be power of 2 for alignment.");

    static constexpr size_t BLOCK_COUNT = ITEM_COUNT - 1;
    static constexpr uintptr_t EMPTY_MASK = 1 << 0;

    // An element in the Bag may be empty. Where this is the case, a
    // freelist is threaded through holes so that the element's slot can be
    // reused. This is encoded as the `hole_ptr` field in the union.
    union MaybeElem
    {
      MaybeElem* hole_ptr;
      Elem item;
    };

    struct alignas(ITEM_COUNT * sizeof(MaybeElem)) Block
    {
      MaybeElem prev;
      MaybeElem data[BLOCK_COUNT];
    };

    // Dummy block to effectively allow pointer arithmetic on nullptr
    // which is undefined behaviour.  So we statically allocate a block
    // to represent the end of the bag.
    inline static Block null_block{};

    // Index of the full dummy block
    // Due to pointer arithmetic with nullptr being undefined behaviour
    // we use a statically allocated null block.
    static constexpr MaybeElem* null_index =
      &(null_block.data[BLOCK_COUNT - 1]);

    /// Mask to access the index component of the pointer to a block.
    static constexpr uintptr_t INDEX_MASK =
      (ITEM_COUNT - 1) * sizeof(MaybeElem);

    /// Pointer into a block.  As the blocks are strongly aligned
    /// the bits 9-3 represent the element in the block, with 0 being
    /// a pointer to the `prev` pointer, and implying the empty block.
    MaybeElem* index;

    // Used to thread a freelist pointer through the bag.
    MaybeElem* next_free;

    /// Takes an index and returns the pointer to the Block
    static Block* get_block(MaybeElem* ptr)
    {
      return snmalloc::pointer_align_down<sizeof(Block), Block>(ptr);
    }

    /// Checks if an index into a block means the block is empty.
    static bool is_first_block_elem(MaybeElem* ptr)
    {
      return ((uintptr_t)ptr & INDEX_MASK) == 0;
    }

    /// Checks if an index into a block means the block has no space.
    static bool is_last_block_elem(MaybeElem* index)
    {
      return ((uintptr_t)index & INDEX_MASK) == INDEX_MASK;
    }

  public:
    Bag<T, U, Alloc>() : index(null_index), next_free(nullptr)
    {
      static_assert(
        sizeof(*this) == sizeof(void*) * 2,
        "Stack should contain only the index and freelist pointer");
    }

    /// Deallocate the linked blocks for this bag.
    void dealloc(Alloc& alloc)
    {
      auto local_block = get_block(index);
      while (local_block != &null_block)
      {
        auto prev = get_block(local_block->prev.hole_ptr);
        alloc.template dealloc<sizeof(Block)>(local_block);
        local_block = prev;
      }
      index = null_index;
    }

    ALWAYSINLINE void remove(Elem* item)
    {
      MaybeElem* hole = (MaybeElem*)item;
      hole->hole_ptr = (MaybeElem*)((uintptr_t)next_free | EMPTY_MASK);
      next_free = hole;
    }

    /// Insert an element into the bag.
    ALWAYSINLINE Elem* insert(Elem item, Alloc& alloc)
    {
      if (next_free != nullptr)
      {
        MaybeElem* prev = next_free->hole_ptr;
        next_free->item = item;
        assert((uintptr_t)prev & EMPTY_MASK);
        MaybeElem* cur = next_free;
        next_free = (MaybeElem*)((uintptr_t)prev & ~EMPTY_MASK);
        return &(cur->item);
      }
      if (!is_last_block_elem(index))
      {
        index++;
        index->item = item;
        return &(index->item);
      }
      return insert_slow(item, alloc);
    }

  private:
    /// Slow path for insert, performs an insert, when allocation is required.
    Elem* insert_slow(Elem item, Alloc& alloc)
    {
      assert(is_last_block_elem(index));

      Block* next = (Block*)alloc.template alloc<sizeof(Block)>();
      assert(((uintptr_t)next) % alignof(Block) == 0);
      next->prev.hole_ptr = index;
      index = &(next->data[0]);
      index->item = item;
      return &(index->item);
    }

    static void step(MaybeElem*& elem)
    {
      elem--;
      if (is_first_block_elem(elem))
      {
        elem = get_block(elem)->prev.hole_ptr;
      }
    }

    static MaybeElem* next_non_empty(MaybeElem* elem)
    {
      while ((elem != Bag::null_index) &&
             ((uintptr_t)elem->hole_ptr & EMPTY_MASK))
      {
        step(elem);
      }
      return elem;
    }

  public:
    class iterator
    {
      friend class Bag;

    public:
      iterator(Bag<T, U, Alloc>* bag) : bag(bag)
      {
        ptr = bag->next_non_empty(bag->index);
      }

      iterator(Bag<T, U, Alloc>* bag, MaybeElem* p) : bag(bag), ptr(p) {}

      iterator operator++()
      {
        step(ptr);
        ptr = next_non_empty(ptr);
        assert(
          (ptr == Bag::null_index) ||
          (((uintptr_t)ptr->hole_ptr & EMPTY_MASK) == 0));

        return *this;
      }

      inline bool operator!=(const iterator& other) const
      {
        return ptr != other.ptr;
      }

      inline bool operator==(const iterator& other) const
      {
        return ptr == other.ptr;
      }

      inline Elem* operator*() const
      {
        return &(ptr->item);
      }
      inline iterator begin()
      {
        return {bag};
      }

      inline iterator end()
      {
        return {bag, Bag::null_index};
      }

    private:
      Bag<T, U, Alloc>* bag;
      MaybeElem* ptr;
    };

    inline iterator begin()
    {
        return {this};
    }

    inline iterator end()
    {
        return {this, Bag::null_index};
    }

  };
} // namespace verona::rt
