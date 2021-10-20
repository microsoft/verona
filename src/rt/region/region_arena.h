// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../object/object.h"
#include "region_base.h"

#include <cstddef>

namespace verona::rt
{
  using namespace snmalloc;

  /**
   * Please see region.h for the full documentation.
   *
   * This is a concrete implementation of a region, specifically one with an
   * arena for bump-allocated objects. This class inherits from RegionBase, but
   * it cannot call any of the static methods in Region.
   *
   * An arena region object has a pointer to a linked list of arenas, where
   * objects are allocated. To ensure merges are fast, we also keep a pointer
   * to the last arena in the linked list.
   *
   * When allocating a new object, we check if the last arena has free space.
   * If so, we allocate within that arena. If not, we allocate a new arena and
   * then allocate the object within the new arena. Note that we do not do
   * first fit or best fit.
   *
   * Note that if the Iso is allocated within an arena, it will still point to
   * the arena region object.
   *
   * Objects that are too large to be allocated within an arena are allocated
   * by snmalloc and placed into the large object ring, a circular linked list
   * of objects accessed via the Object::next pointer. This ring mixes both
   * trivial and non-trivial objects. Since the iso object may not be in the
   * large object ring, we need a pointer to the last object in the ring, to
   * ensure merges are fast. If the iso object is in the large object ring,
   * then it must be in the last position, so it can point to the region
   * metadata object.
   **/
  class RegionArena : public RegionBase
  {
  public:
    // Need to foward declare so Arena can declare a friend class.
    template<IteratorType type>
    class iterator;

  private:
    friend class Region;
    friend class RegionTrace;
    friend class RegionRc;

    /**
     * An Arena is a large block of pre-allocated memory. It has an overhead of
     * four pointers: the next Arena in the linked list, and three pointers to
     * keep track of where objects are allocated. The next pointers of all
     * objects inside an arena are set to nullptr. An initialized arena is
     * guaranteed to have at least one object.
     *
     * Trivial objects (ie. those with no destructor, no finaliser and no iso
     * fields) are allocated from the beginning of the arena, starting at
     * `objects_begin`. `objects_end` points to the first byte after the last
     * object, i.e. the place where the next object will be allocated. `objects`
     * begin points to the start of the allocation, rather then the Object*
     * that stores the header before it.
     *
     * Non-trivial objects are allocated from the end of the arena.
     * `non_trivial_end` points past the end of the arena and
     * `non_trivial_begin` points to the first non-trivial object.
     * `non_trivial_begin` points to the start of the header of the first
     *object.
     *
     * Note that certain operations require the bottom `MIN_ALLOC_BITS` to be
     * free, so we need to ensure all objects allocated within an arena are
     * properly aligned. This may involve extra padding in the "header" of an
     * Arena object, and also rounding up object sizes.
     *
     *                       +-------------------+
     *                       | next arena ---------> ...
     *                       | objects_end       |
     *                       | non_trivial_begin |
     *                       | non_trivial_end   |
     *                       |===================|
     *    objects_begin ---> | object_1          |
     *                       +-------------------+
     *                       | ...               |
     *                       +-------------------+
     *                       | object_n          |
     *                       +~~~~~~~~~~~~~~~~~~~+
     *      objects_end ---> | free space        |
     *                       |                   |
     *                       +~~~~~~~~~~~~~~~~~~~+
     * non_trivial_begin --> | non_trivial_m     |
     *                       +-------------------+
     *                       | ...               |
     *                       +-------------------+
     *                       | non_trivial_1     |
     *                       +-------------------+
     *   non_trivial_end --->
     *
     * We can iterate over objects by starting from `objects_begin`, moving the
     * pointer by the size of the current object, until we reach `objects_end`.
     * We iterate from the first allocated object to the last allocated object.
     *
     * We can iterate over non-trivial objects by starting from
     * `non_trivial_begin`, moving the pointer by the size of the current
     * object, until we reach `non_trivial_end`. We iterate from the last
     * allocated object to the first allocated object.
     *
     * We can calculate the remaining free space by taking the difference of
     * `non_trivial_begin` and `objects_end`.
     **/
    class Arena
    {
      template<IteratorType type>
      friend class RegionArena::iterator;

    public:
      static constexpr size_t SIZE = 1024 * 1024 - 4 * sizeof(uintptr_t);

      /**
       * Pointer to next arena in the linked list.
       **/
      Arena* next;

    private:
      /**
       * Pointer to one past the last allocated object, i.e. where the next
       * object will be allocated, assuming sufficient space.
       **/
      std::byte* objects_end;

      /**
       * Pointer to the first allocated non-trivial object.
       **/
      std::byte* non_trivial_begin;

      /**
       * Pointer to the byte after the Arena. We technically don't need to
       * store this pointer, but we have some space because `objects_begin`
       * needs to be aligned.
       **/
      std::byte* non_trivial_end;

      /**
       * Where objects will actually be allocated.
       **/
      alignas(Object::ALIGNMENT) std::byte objects_begin[SIZE];

    public:
      Arena()
      : next(nullptr),
        objects_end(objects_begin),
        non_trivial_begin(objects_begin + SIZE),
        non_trivial_end(non_trivial_begin)
      {
        assert(free_space() == SIZE);
      }

      inline size_t free_space() const
      {
        assert(debug_invariant());
        std::ptrdiff_t diff = non_trivial_begin - objects_end;
        return (size_t)diff;
      }

      /**
       * Allocates an object of type `desc` within the current arena.
       * `sz` is how much space the object will take up, accounting for padding
       * to ensure alignment. Note that `desc->size` is the actual size of the
       * object, and may have been rounded up to obtain `sz`.
       *
       * Since the memory has already been pre-allocated, we just need to
       * update a few pointers.
       *
       * Returns a pointer to where the object should be constructed.
       **/
      Object* alloc_obj(const Descriptor* desc, size_t sz)
      {
        assert(debug_invariant());
        assert(free_space() >= sz);

        void* p = nullptr;

        if (Object::is_trivial(desc))
        {
          p = objects_end;
          objects_end += sz;
        }
        else
        {
          non_trivial_begin -= sz;
          p = non_trivial_begin;
        }

        auto o = Object::register_object(p, desc);
        o->init_next(nullptr);

        assert(debug_invariant());
        return o;
      }

    private:
      bool debug_invariant() const
      {
        bool objects_ptrs = objects_begin <= objects_end;
        bool non_trivial_ptrs = non_trivial_begin <= non_trivial_end;
        bool no_overlap = (non_trivial_begin - objects_end) >= 0;
        auto alignment1 = Object::debug_is_aligned(objects_begin);
        auto alignment2 = Object::debug_is_aligned(objects_end);
        auto alignment3 = Object::debug_is_aligned(non_trivial_begin);
        auto alignment4 = Object::debug_is_aligned(non_trivial_end);
        return objects_ptrs && non_trivial_ptrs && no_overlap && alignment1 &&
          alignment2 && alignment3 && alignment4;
      }
    };
    static_assert(sizeof(Arena) == 1024 * 1024 * sizeof(std::byte));

    /**
     * Pointer to the linked list of arenas where objects are allocated in.
     * May be null, if all of the objects are in the large object ring.
     **/
    Arena* first_arena;

    /**
     * Pointer to the last arena in the linked list of arenas.
     * May be null, if all of the objects are in the large object ring.
     **/
    Arena* last_arena;

    /**
     * Large object ring, i.e. the circular linked list of large objects that
     * don't fit into arenas. This pointer is for the "last" (possibly iso)
     * object in the list, to make certain operations faster. We use the
     * Object::next pointer for the "first" object in the list.
     **/
    Object* last_large;

    RegionArena()
    : RegionBase(),
      first_arena(nullptr),
      last_arena(nullptr),
      last_large(nullptr)
    {
      init_next(this);
    }

    static const Descriptor* desc()
    {
      static constexpr Descriptor desc = {
        vsizeof<RegionArena>, nullptr, nullptr, nullptr};

      return &desc;
    }

  public:
    inline static RegionArena* get(Object* o)
    {
      assert(o->debug_is_iso());
      assert(is_arena_region(o->get_region()));
      return (RegionArena*)o->get_region();
    }

    inline static bool is_arena_region(Object* o)
    {
      return o->is_type(desc());
    }

    /**
     * Creates a new arena region by allocationg Object `o` of type `desc`. The
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
      void* p = Object::register_object(
        alloc.alloc<vsizeof<RegionArena>>(), RegionArena::desc());
      RegionArena* reg = new (p) RegionArena();

      // o might be allocated in the arena or the large object ring.
      Object* o = reg->alloc_internal<size>(alloc, desc);
      assert(Object::debug_is_aligned(o));

      o->init_iso();
      o->set_region(reg);
      assert(
        reg->last_large != nullptr ?
          reg->last_large->get_next_any_mark() == reg :
          true);

      return o;
    }

    /**
     * Allocates an object `o` of type `desc` in the region represented by the
     * Iso object `in`. `o` will be allocated in an arena or the large object
     * ring. Returns a pointer to `o`.
     *
     * The default template parameter `size = 0` is to avoid writing two
     * definitions which differ only in one line. This overload works because
     * every object must contain a descriptor, so 0 is not a valid size.
     **/
    template<size_t size = 0>
    static Object* alloc(Alloc& alloc, Object* in, const Descriptor* desc)
    {
      RegionArena* reg = get(in);
      Object* o = reg->alloc_internal<size>(alloc, desc);
      assert(Object::debug_is_aligned(o));
      return o;
    }

    /**
     * Insert the Object `o` into the RememberedSet of `into`'s region.
     *
     * If ownership of a reference count is being transfered to the region,
     * pass the template argument `transfer = YesTransfer`.
     *
     * TODO(region): Does this make sense for a RegionArena? Or do we want
     * to keep precise refcounts? Should this method be exposed through the
     * Region class and automatically do the dispatch, or do we want to force
     * the caller to cast?
     **/
    template<TransferOwnership transfer = NoTransfer>
    static void insert(Alloc& alloc, Object* into, Object* o)
    {
      assert(o->debug_is_immutable() || o->debug_is_cown());
      RegionArena* reg = get(into);

      Object::RegionMD c;
      o = o->root_and_class(c);
      reg->RememberedSet::insert<transfer>(alloc, o);
    }

    /**
     * Merges `o`'s region into `into`'s region. Both regions must be separate
     * and be the same kind of region, e.g. two trace regions.
     *
     * TODO(region): how to handle merging different types of regions?
     **/
    static void merge(Alloc& alloc, Object* into, Object* o)
    {
      assert(o->debug_is_iso());
      RegionArena* reg = get(into);
      RegionBase* other = o->get_region();
      assert(reg != other);

      if (is_arena_region(other))
        reg->merge_internal((RegionArena*)other);
      else
        assert(0);

      // Clear the iso bit on `o`, if it's inside an arena. Otherwise, it's in
      // the large object ring and pointing to some other object.
      size_t sz = snmalloc::bits::align_up(o->size(), Object::ALIGNMENT);
      if (sz <= Arena::SIZE)
        o->init_next(nullptr);

      // Merge the ExternalRefTable and RememberedSet.
      reg->ExternalReferenceTable::merge(alloc, other);
      reg->RememberedSet::merge(alloc, other);

      // Now we can deallocate the other region's metadata object.
      other->dealloc(alloc);
    }

    /**
     * Swap the Iso (root) Object of a region, `prev`, with another Object
     * within that region, `next`.
     **/
    static void swap_root(Object* prev, Object* next)
    {
      assert(prev != next);
      assert(prev->debug_is_iso());
      assert(next->debug_is_mutable());
      assert(prev->get_region() != next);

      RegionArena* reg = get(prev);
      reg->swap_root_internal(prev, next);
    }

  private:
    inline void append(Object* hd)
    {
      append(hd, hd);
    }

    /**
     * Inserts the object `hd` into the large object ring, right after the
     * region metadata object. `tl` is used for merging two rings; if there is
     * only one ring, then hd == tl.
     **/
    inline void append(Object* hd, Object* tl)
    {
      Object* p = get_next();
      tl->init_next(p);
      set_next(hd);
      if (last_large == nullptr)
        last_large = tl;
    }

    /**
     * Allocate an object of type `desc` in the region. Returns a pointer to
     * that object.
     *
     * If the object is too large to fit in an arena, new memory is allocated
     * and the object is added to the large object ring.
     *
     * Otherwise, we check if the last arena has space. If so, the object is
     * allocated there. If not, we allocate a new arena.
     *
     * TODO(region): For now, we guarantee constant-time allocation and accept
     * that we will have fragmentation. Later, we could try other strategies,
     * e.g. first fit or best fit.
     **/
    template<size_t size = 0>
    Object* alloc_internal(Alloc& alloc, const Descriptor* desc)
    {
      assert((size == 0) || (desc->size == size));

      auto sz = size == 0 ? desc->size : size;
      if (sz > Arena::SIZE)
      {
        // Allocate object.
        void* p = nullptr;
        if constexpr (size == 0)
          p = alloc.alloc(desc->size);
        else
          p = alloc.alloc<size>();

        auto o = Object::register_object(p, desc);

        // Add to large object ring
        append(o);

        return o;
      }

      // If we don't have an arena, or the arena does not have enough space,
      // allocate a new arena.
      if (last_arena == nullptr || last_arena->free_space() < sz)
      {
        void* p = alloc.alloc<sizeof(Arena)>();
        Arena* a = new (p) Arena();

        if (last_arena == nullptr)
        {
          first_arena = a;
          last_arena = a;
        }
        else
        {
          last_arena->next = a;
          last_arena = a;
        }
        assert(last_arena->next == nullptr);
      }

      // Allocate object within that arena.
      return last_arena->alloc_obj(desc, sz);
    }

    void merge_internal(RegionArena* other)
    {
      // Merge arena linked lists.
      if (last_arena == nullptr)
      {
        assert(first_arena == nullptr);
        first_arena = other->first_arena;
        last_arena = other->last_arena;
      }
      else
      {
        if (other->first_arena != nullptr)
        {
          assert(other->last_arena != nullptr);
          last_arena->next = other->first_arena;
          last_arena = other->last_arena;
        }
      }

      // Merge large object ring.
      Object* head = other->get_next();
      if (head != other)
        append(head, other->last_large);

      assert(last_arena != nullptr ? last_arena->next == nullptr : true);
      assert(
        last_large != nullptr ? last_large->get_next_any_mark() == this : true);
    }

    void swap_root_internal(Object* oroot, Object* nroot)
    {
      assert(debug_is_in_region(nroot));
      size_t oroot_size =
        snmalloc::bits::align_up(oroot->size(), Object::ALIGNMENT);
      size_t nroot_size =
        snmalloc::bits::align_up(nroot->size(), Object::ALIGNMENT);

      if (oroot_size <= Arena::SIZE)
      {
        // Old root is inside an arena, so we set its next to nullptr.
        oroot->init_next(nullptr);
      }
      else
      {
        // Old root is in the large object ring.
        assert(oroot == last_large);
        if (nroot_size <= Arena::SIZE)
        {
          // Clear the iso bit on the old root.
          oroot->init_next(this);
        }
      }

      // New root is in the large object ring, need to move it to the last
      // position in the ring. Don't do anything if it's already last.
      if (nroot != last_large && nroot_size > Arena::SIZE)
      {
        Object* x = get_next();
        Object* y = nroot->get_next();
        last_large->init_next(x);
        set_next(y);
        last_large = nroot;
      }

      // Doesn't matter where the new root is, we have to set its iso bit and
      // have it point to the region metadata object.
      nroot->init_iso();
      nroot->set_region(this);

      assert(
        last_large != nullptr ? last_large->get_next_any_mark() == this : true);
    }

    /**
     * Release and deallocate all objects within the region represented by the
     * Iso Object `o`.
     *
     * Note: this does not release subregions. Use Region::release instead.
     **/
    void release_internal(Alloc& alloc, Object* o, ObjectStack& collect)
    {
      assert(o->debug_is_iso());
      // Don't trace or finalise o, we'll do it when looping over the large
      // object ring or the arena list.

      Systematic::cout() << "Region release: arena region: " << o
                         << Systematic::endl;

      // Clean up all the non-trivial objects, by running the finaliser and
      // destructor, and collecting iso regions.
      // Finalisers must provide all the isos of the current object that will
      // need collecting.  The language must guarantee that there are no isos
      // left in this object that haven't been added to collect.  Read-only
      // views of objects during finalization are the easiest way to guarantee
      // this.
      for (auto it = begin<NonTrivial>(); it != end<NonTrivial>(); ++it)
      {
        (*it)->finalise(o, collect);
      }

      // Destructors can invalidate the object's state, so all finalisers must
      // run before any destructor runs, i.e. two separate passes are required.
      for (auto it = begin<NonTrivial>(); it != end<NonTrivial>(); ++it)
      {
        (*it)->destructor();
      }

      // Now we can deallocate large object ring.
      Object* p = get_next();
      while (p != this)
      {
        Object* q = p->get_next_any_mark();
        p->dealloc(alloc);
        p = q;
      }

      // Deallocate arenas.
      Arena* arena = first_arena;
      while (arena != nullptr)
      {
        Arena* q = arena->next;
        alloc.dealloc<sizeof(Arena)>(arena);
        arena = q;
      }

      // Sweep the RememberedSet, to ensure destructors are called.
      RememberedSet::sweep(alloc);

      // Deallocate RegionArena
      // Don't need to deallocate `o`, since it was part of the arena or ring.
      dealloc(alloc);
    }

  public:
    template<IteratorType type = AllObjects>
    class iterator
    {
      friend class RegionArena;

      static_assert(
        type == Trivial || type == NonTrivial || type == AllObjects);

      iterator(RegionArena* r) : reg(r), arena(r->first_arena), ptr(nullptr)
      {
        init_for_arena_or_ring();
      }

      iterator(RegionArena* r, Arena* a, Object* p) : reg(r), arena(a), ptr(p)
      {}

    public:
      iterator operator++()
      {
        if (arena != nullptr)
        {
          // Currently iterating through an arena.
          ptr = next_in_arena();
          if (ptr == nullptr)
          {
            // Go to the next arena or large object ring.
            arena = arena->next;
            init_for_arena_or_ring();
          }
        }
        else
        {
          // Currenty iterating through the large object ring.
          ptr = next_in_ring(ptr);
        }
        return *this;
      }

      inline bool operator!=(const iterator& other) const
      {
        assert(reg == other.reg);
        return ptr != other.ptr;
      }

      inline Object* operator*() const
      {
        return ptr;
      }

    private:
      RegionArena* reg;
      Arena* arena;
      /// ptr points to the object after the header, as this is the pointer
      /// used most commonly in the runtime.
      Object* ptr;

      /**
       * Within the current arena, return a pointer to the next object to be
       * iterated. Return nullptr if we reach the end.
       **/
      inline Object* next_in_arena() const
      {
        assert(arena->debug_invariant());
        size_t sz = snmalloc::bits::align_up(ptr->size(), Object::ALIGNMENT);
        // Get actual end of the object, that is,
        // q points to the start of the header of the next object.
        std::byte* q = ptr->real_start() + sz;
        if constexpr (type == Trivial)
        {
          assert(q > arena->objects_begin && q <= arena->objects_end);

          // We have not yet reached the end, so q is valid.
          if (q != arena->objects_end)
            return Object::object_start(q);
        }
        else if constexpr (type == NonTrivial)
        {
          assert(q > arena->non_trivial_begin && q <= arena->non_trivial_end);

          // We have not yet reached the end, so q is valid.
          if (q != arena->non_trivial_end)
            return Object::object_start(q);
        }
        else if constexpr (type == AllObjects)
        {
          assert(
            (q > arena->objects_begin && q <= arena->objects_end) ||
            (q > arena->non_trivial_begin && q <= arena->non_trivial_end));

          // We have not yet reached either end, so q is valid.
          if (q != arena->objects_end && q != arena->non_trivial_end)
            return Object::object_start(q);

          // We reached the end of trivial objects and there are non-trivial
          // objects to iterate over.
          if (
            q == arena->objects_end &&
            arena->non_trivial_begin != arena->non_trivial_end)
            return Object::object_start(arena->non_trivial_begin);
        }
        return nullptr;
      }

      /**
       * Starting from the current `arena`, set `ptr` to the first
       * "appropriate" Object.
       *
       * If no object can be found in the arena list, then we try the large
       * object ring. If no objects are left, then `ptr` is set to nullptr.
       *
       * This method updates both the current `arena` and Object `ptr`.
       **/
      inline void init_for_arena_or_ring()
      {
        // Search through the arena list for an object.
        ptr = first_in_arena_list();

        // Didn't find anything in the arenas, so try the large object ring.
        if (ptr == nullptr)
          ptr = next_in_ring(reg);
        return;
      }

      /**
       * Starting from the current `arena`, return a pointer to the first
       * appropriate Object. If no object is found, return nullptr.
       *
       * Note that an arena is guaranteed to have at least one object.
       *
       * This method updates the current `arena` pointer, as it may need to
       * traverse the arena list.
       **/
      inline Object* first_in_arena_list()
      {
        while (arena != nullptr)
        {
          assert(
            arena->objects_begin < arena->objects_end ||
            arena->non_trivial_begin < arena->non_trivial_end);
          assert(arena->debug_invariant());
          if constexpr (type == Trivial || type == AllObjects)
          {
            if (arena->objects_begin != arena->objects_end)
              // objects_begin points to header of first object.
              // we return the actually Object*.
              return Object::object_start(arena->objects_begin);
          }
          if constexpr (type == NonTrivial || type == AllObjects)
          {
            if (arena->non_trivial_begin != arena->non_trivial_end)
              return Object::object_start(arena->non_trivial_begin);
          }
          // Every arena contains at least one object.
          if constexpr (type == AllObjects)
            assert(0);
          arena = arena->next;
        }
        return nullptr;
      }

      /**
       * Starting from object `p` in the large object ring, return a pointer to
       * the next appropriate Object. If no object is found, return nullptr.
       *
       * `p` may be `reg`, which means starting at the beginning of the ring.
       **/
      inline Object* next_in_ring(Object* p) const
      {
        Object* q = p->get_next_any_mark();
        while (q != reg)
        {
          bool cond;
          if constexpr (type == Trivial)
            cond = q->is_trivial();
          else if constexpr (type == NonTrivial)
            cond = !q->is_trivial();
          else
            cond = true;

          if (cond)
            return q;
          q = q->get_next_any_mark();
        }
        return nullptr;
      }
    };

    template<IteratorType type = AllObjects>
    inline iterator<type> begin()
    {
      return {this};
    }

    template<IteratorType type = AllObjects>
    inline iterator<type> end()
    {
      return {this, nullptr, nullptr};
    }

  private:
    bool debug_is_in_region(Object* o)
    {
      for (auto p : *this)
      {
        if (p == o)
          return true;
      }
      return false;
    }
  };
} // namespace verona::rt
