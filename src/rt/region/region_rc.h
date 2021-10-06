// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../object/object.h"
#include "region_arena.h"
#include "region_base.h"

namespace verona::rt
{
  using namespace snmalloc;

  /**
   * A Region Vector is used to track all allocations in a region. It is a
   * vector-like data structure optimised for constant time insertion and
   * removal.
   *
   * Removal is O(1) because a removed item will leave a hole in the vector
   * rather than shifting remaining items down. To reduce fragmentation which
   * can occur with deallocation churn, the region vector maintains a freelist
   * which is threaded through the holes left in the vector.  Insertion of new
   * items will first query the freelist to see if a hole can be reused,
   * otherwise items are bump allocated.
   *
   * To maintain an internal freelist with no additional space requirements, the
   * item `T` must be at least 1 machine word in size.
   */
  template<class T, class Alloc>
  class RegionVector
  {
    static constexpr size_t ITEM_COUNT = 32;
    static_assert(
      snmalloc::bits::next_pow2_const(ITEM_COUNT) == ITEM_COUNT,
      "Should be power of 2 for alignment.");

    static constexpr size_t BLOCK_COUNT = ITEM_COUNT - 1;
    static constexpr uintptr_t EMPTY_MASK = 1 << 0;

    struct alignas(ITEM_COUNT * sizeof(T)) Block
    {
      T prev;
      T data[BLOCK_COUNT];
    };

  private:
    // Dummy block to effectively allow pointer arithmetic on nullptr
    // which is undefined behaviour.  So we statically allocate a block
    // to represent the end of the vec.
    inline static Block null_block{};

    // Index of the full dummy block
    // Due to pointer arithmetic with nullptr being undefined behaviour
    // we use a statically allocated null block.
    static constexpr T* null_index = &(null_block.data[BLOCK_COUNT - 1]);

    /// Mask to access the index component of the pointer to a block.
    static constexpr uintptr_t INDEX_MASK = (ITEM_COUNT - 1) * sizeof(T);

    /// Pointer into a block.  As the blocks are strongly aligned
    /// the bits 9-3 represent the element in the block, with 0 being
    /// a pointer to the `prev` pointer, and implying the empty block.
    T* index;

    // Used to thread a freelist pointer through the vec.
    T* next_free;

    /// Takes an index and returns the pointer to the Block
    static Block* get_block(T* ptr)
    {
      return snmalloc::pointer_align_down<sizeof(Block), Block>(ptr);
    }

    /// Checks if an index into a block means the block is empty.
    static bool is_empty(T* ptr)
    {
      return ((uintptr_t)ptr & INDEX_MASK) == 0;
    }

    /// Checks if an index into a block means the block has no space.
    static bool is_full(T* index)
    {
      return ((uintptr_t)index & INDEX_MASK) == INDEX_MASK;
    }

  public:
    RegionVector<T, Alloc>() : index(null_index), next_free(nullptr)
    {
      static_assert(
        sizeof(*this) == sizeof(void*) * 2,
        "Stack should contain only the index and freelist pointer");
    }

    /// Deallocate the linked blocks for this vec.
    void dealloc(Alloc& alloc)
    {
      auto local_block = get_block(index);
      while (local_block != &null_block)
      {
        auto prev_ptr = *(T**)&(local_block->prev);
        auto prev = get_block(prev_ptr);
        alloc.template dealloc<sizeof(Block)>(local_block);
        local_block = prev;
      }
      index = null_index;
    }

    ALWAYSINLINE void remove(T* index)
    {
      *(uintptr_t*)index = ((uintptr_t)next_free | EMPTY_MASK);
      next_free = index;
    }

    /// returns true if this vector is empty
    ALWAYSINLINE bool empty()
    {
      return index == null_index;
    }

    /// Return the top element of the vector without removing it.
    ALWAYSINLINE T* peek()
    {
      assert(!empty());
      return index;
    }

    /// Push an element to the back of the vector.
    ALWAYSINLINE T* push(T item, Alloc& alloc)
    {
      if (next_free != nullptr)
      {
        uintptr_t prev = *(uintptr_t*)next_free;
        *next_free = item;
        T* cur = next_free;
        next_free = (T*)(prev & ~EMPTY_MASK);
        return cur;
      }
      if (!is_full(index))
      {
        index++;
        *index = item;
        return index;
      }
      return push_slow(item, alloc);
    }

  private:
    /// Slow path for push, performs a push, when allocation is required.
    T* push_slow(T item, Alloc& alloc)
    {
      assert(is_full(index));

      Block* next = (Block*)alloc.template alloc<sizeof(Block)>();
      assert(((uintptr_t)next) % alignof(Block) == 0);
      T** prev = (T**)&next->prev;
      *prev = index;
      index = &(next->data[0]);
      *index = item;
      return index;
    }

  public:
    class iterator
    {
      friend class RegionVector;

    public:
      iterator(RegionVector<T, Alloc>* vec) : vec(vec)
      {
        // If the vec is empty then there is nothing to iterate over.
        if (vec->empty())
        {
          ptr = nullptr;
          return;
        }

        ptr = vec->peek();
      }

      iterator(RegionVector<T, Alloc>* vec, T* p) : vec(vec), ptr(p) {}

      iterator operator++()
      {
        ptr--;
        if (ptr != vec->null_index)
        {
          if (!vec->is_empty(ptr))
          {
            next_non_empty();
            return *this;
          }
          ptr = *(T**)&(vec->get_block(ptr)->prev);
          if (ptr != vec->null_index)
          {
            next_non_empty();
            return *this;
          }
        }
        ptr = nullptr;
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

      inline ObjectCount* operator*() const
      {
        return ptr;
      }
      inline iterator begin()
      {
        return {vec};
      }

      inline iterator end()
      {
        return {vec, nullptr};
      }

    private:
      inline void next_non_empty()
      {
        uintptr_t objptr = *(uintptr_t*)ptr;
        while (objptr & EMPTY_MASK && !vec->is_empty(ptr))
        {
          ptr--;
          objptr = *(uintptr_t*)ptr;
        };
      }

      RegionVector<T, Alloc>* vec;
      T* ptr;
    };
  };

  /**
   * Please see region.h for the full documentation.
   *
   * This is a concrete implementation of a region, specifically one with
   * reference counting. This class inherits from RegionBase, but it cannot call
   * any of the static methods in Region.
   *
   * In a rc region, all objects are tracked using "region vector"s, where each
   * element is an `ObjectCount`: a pair containing a pointer to the object and
   * its current reference count. The region vector is each time an object is
   * allocated or deallacted.
   *
   * If an object is trivial (ie. it has no finaliser, no destructor and no
   * subregions), it is tracked using the trivial vector. Conversely,
   *non-trivial objects are tracked with the non-trivial vector.
   **/
  class RegionRc : public RegionBase
  {
    friend class Freeze;
    friend class Region;

  private:
    RegionVector<ObjectCount, Alloc> trivial_counts{};
    RegionVector<ObjectCount, Alloc> non_trivial_counts{};

    // Memory usage in the region.
    size_t current_memory_used = 0;

    // Compact representation of previous memory used as a sizeclass.
    snmalloc::sizeclass_t previous_memory_used = 0;

    // Stack of stack based entry points into the region.
    StackThin<Object, Alloc> additional_entry_points{};

    RegionRc() : RegionBase() {}

    static const Descriptor* desc()
    {
      static constexpr Descriptor desc = {
        vsizeof<RegionRc>, nullptr, nullptr, nullptr};

      return &desc;
    }

  public:
    inline static RegionRc* get(Object* o)
    {
      assert(o->debug_is_iso());
      assert(is_rc_region(o->get_region()));
      return (RegionRc*)o->get_region();
    }

    inline static bool is_rc_region(Object* o)
    {
      return o->is_type(desc());
    }

    inline RegionVector<ObjectCount, Alloc>* get_trivial_vec()
    {
      return &trivial_counts;
    }

    inline RegionVector<ObjectCount, Alloc>* get_non_trivial_vec()
    {
      return &non_trivial_counts;
    }

    /**
     * Creates a new rc region by allocating Object `o` of type `desc`. The
     * object is initialised as the Iso object for that region, and points to a
     * newly created Region metadata object. Returns a pointer to `o`.
     *
     * The default template parameter `size = 0` is to avoid writing two
     * definitions which differ only in one line. This overload works because
     * every object must contain a descriptor, so 0 is not a valid size.
     **/
    template<size_t size = 0>
    static Object* create(Alloc& alloc, const Descriptor* desc)
    {
      void* p = alloc.alloc<vsizeof<RegionRc>>();
      Object* o = Object::register_object(p, RegionRc::desc());
      auto reg = new (o) RegionRc();
      reg->use_memory(desc->size);

      if constexpr (size == 0)
        p = alloc.alloc(desc->size);
      else
        p = alloc.alloc<size>();
      o = Object::register_object(p, desc);

      reg->init_next(o);
      o->init_iso();
      o->set_region(reg);

      assert(Object::debug_is_aligned(o));
      return o;
    }

    /**
     * Allocates an object `o` of type `desc` in the region represented by the
     * Iso object `in`, and adds it to the appropriate ring. Returns a pointer
     * to `o`.
     *
     * The default template parameter `size = 0` is to avoid writing two
     * definitions which differ only in one line. This overload works because
     * every object must contain a descriptor, so 0 is not a valid size.
     **/
    template<size_t size = 0>
    static Object* alloc(Alloc& alloc, Object* in, const Descriptor* desc)
    {
      assert((size == 0) || (size == desc->size));
      RegionRc* reg = get(in);

      assert(reg != nullptr);

      void* p = nullptr;
      if constexpr (size == 0)
        p = alloc.alloc(desc->size);
      else
        p = alloc.alloc<size>();

      auto o = (Object*)Object::register_object(p, desc);
      assert(Object::debug_is_aligned(o));

      // Add to the region vector and create a back pointer.
      ObjectCount* oc = nullptr;
      if (in->is_trivial())
        oc = reg->get_trivial_vec()->push({o, 1}, alloc);
      else
        oc = reg->get_non_trivial_vec()->push({o, 1}, alloc);
      o->set_rv_index(oc);

      // GC heuristics.
      reg->use_memory(desc->size);
      return o;
    }

    static ObjectCount* debug_get_rv_index(Object* o)
    {
      return o->get_rv_index();
    }

    static uintptr_t debug_get_ref_count(Object* o)
    {
      ObjectCount* oc = (ObjectCount*)o->get_rv_index();
      return oc->count;
    }

  private:
    void use_memory(size_t size)
    {
      current_memory_used += size;
    }
  };

} // namespace verona::rt