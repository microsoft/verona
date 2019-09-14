// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace verona::rt
{
  /**
   * Robinhood hashmap from Key(Object*) to Value. The Entry is either Object*,
   * using hashmap as a set, or pair{Object*, Value}.
   *
   * The key (obtained from `key_of`) has to be Object*, because the last few
   * bits (normally zero due pointer alignment) are used to encode marking
   * status (MARK) and distance-to-initial-bucket (DIB).
   */
  template<typename Entry, size_t& key_of(Entry*)>
  class PtrKeyHashMap
  {
  private:
    // Number of elements present and size of the array.
    size_t count = 0;
    Entry* set = nullptr;

    // Compact representation of the allocated size of the set as a bit count.
    uint8_t size_bits = 0;

    static constexpr uint8_t INITIAL_SIZE_BITS = 3;
    static constexpr uint8_t MARK = 1 << (MIN_ALLOC_BITS - 1);
    static constexpr uint8_t DIB_MAX = MARK - 1;
    static constexpr size_t POINTER_MASK = ~(((size_t)1 << MIN_ALLOC_BITS) - 1);

    // Both the Object and hashmap implementations steal some of the bottom
    // bits of an Object* pointer. The number of bits used may not be the
    // same, but we need to reserve the same number of bits and not use those
    // bits for addressing, which affects Object alignment.
    static_assert(~(POINTER_MASK | DIB_MAX | MARK) == 0);
    static_assert(~(POINTER_MASK | Object::MASK) == 0);

    inline static Object* get_pointer(size_t p)
    {
      // The Object* returned will retain its RememberedSet mark bit, but not
      // its dib.
      return (Object*)(p & ~(size_t)DIB_MAX);
    }

    void grow(Alloc* alloc)
    {
      // Decide if we should grow or not. Grow at 75% capacity.
      size_t size = get_size();
      size_t grow_threshold = (size * 3) / 4;

      if (count < grow_threshold)
        return;

      Entry* old_set = set;
      size_t old_size = size;
      count = 0;
      assert(size_bits < UINT8_MAX);
      size_bits =
        (size_bits > 0) ? (uint8_t)(size_bits + 1) : INITIAL_SIZE_BITS;
      size = get_size();

      set = (Entry*)alloc->alloc<YesZero>(size * sizeof(Entry));

      if (old_set != nullptr)
      {
        for (size_t index = 0; index < old_size; index++)
        {
          auto entry = &old_set[index];
          auto& key = key_of(entry);
          if (key != 0)
          {
            key = (size_t)get_unmarked_pointer(key);
            size_t dummy;
            insert(alloc, *entry, dummy);
          }
        }

        alloc->dealloc(old_set, old_size * sizeof(size_t*));
      }
    }

    void shrink(Alloc* alloc, size_t marked)
    {
      Entry* old_set = set;
      size_t old_size = get_size();
      count = 0;

      size_bits = marked > 3 ?
        (uint8_t)snmalloc::bits::next_pow2_bits(marked << 1) :
        INITIAL_SIZE_BITS;

      size_t size = get_size();
      assert(size > 0);

      set = (Entry*)alloc->alloc<YesZero>(size * sizeof(Entry));

      if (old_size != 0)
      {
        for (size_t index = 0; index < old_size; index++)
        {
          auto entry = &old_set[index];
          auto& key = key_of(entry);

          if ((key & MARK) != 0)
          {
            key = (size_t)get_unmarked_pointer(key);
            size_t dummy;
            insert(alloc, *entry, dummy);
          }
          else if (key != 0)
          {
            entry->~Entry();
          }
        }

        alloc->dealloc(old_set, old_size * sizeof(Entry));
      }
    }

    inline size_t get_dib(size_t size, size_t index, size_t key)
    {
      size_t dib = key & DIB_MAX;

      if (dib == DIB_MAX)
      {
        // If we've encoded the maximum DIB, it could be an overflow.
        // Recalculate the DIB.
        size_t mask = size - 1;
        dib = (index + size -
               (verona::rt::bits::hash(get_unmarked_pointer(key)) & mask)) &
          mask;
      }

      return dib;
    }

    inline void set_entry(size_t index, Entry& entry, size_t dib)
    {
      // When entry is passed in, it may or may not have a mark bit, but it
      // must have no dib. If the DIB is greater than the maximum DIB, encode
      // it as the maximum DIB. We will recalculate the real DIB when we
      // fetch it.
      auto& key = key_of(&entry);
      key = key | (dib < DIB_MAX ? dib : DIB_MAX);
      set[index] = std::move(entry);
    }

    inline size_t get_size()
    {
      return (size_t)1 << size_bits;
    }

    /**
     * This iterator enables range-based for-loop, traversing each non-empty
     * Entry in the map.
     */
    class Iterator
    {
    private:
      PtrKeyHashMap* map;
      size_t i;

    public:
      Iterator(PtrKeyHashMap* map_, size_t i_) : map{map_}, i{i_} {}

      Entry& operator*()
      {
        return map->set[i];
      }

      Entry* operator->()
      {
        return &map->set[i];
      }

      Iterator& operator++()
      {
        assert(map->count > 0);

        auto size = map->get_size();
        while (true)
        {
          i++;
          if (i == size)
            break;
          if (key_of(&map->set[i]) != 0)
            break;
        }

        return *this;
      }

      size_t get_index()
      {
        return i;
      }

      bool operator!=(const Iterator& it) const
      {
        return i != it.i;
      }

      bool operator==(const Iterator& it) const
      {
        return i == it.i;
      }
    };

  public:
    static PtrKeyHashMap* create()
    {
      auto r = ThreadAlloc::get()->alloc<sizeof(PtrKeyHashMap)>();
      return new (r) PtrKeyHashMap();
    }

    static Object* get_unmarked_pointer(size_t p)
    {
      assert(p != 0);
      // The Object* returned has no mark bits or dib.
      return (Object*)(p & POINTER_MASK);
    }

    Iterator begin()
    {
      Iterator i{this, 0};
      if (count == 0)
      {
        return i;
      }
      if (key_of(&set[0]) != 0)
      {
        return i;
      }
      ++i;
      return i;
    }

    Iterator end()
    {
      if (count == 0)
      {
        return {this, 0};
      }
      size_t size = get_size();
      return {this, size};
    }

    void mark_slot(size_t index, size_t& marked)
    {
      assert(index < get_size());
      auto& key = key_of(&set[index]);
      assert(key != 0);

      if ((key & MARK) == 0)
      {
        key = key | MARK;
        marked++;
      }
    }

    template<bool require_destructor = true>
    inline void dealloc(Alloc* alloc)
    {
      if (size_bits > 0)
      {
        if (require_destructor)
        {
          auto size = get_size();
          for (size_t i = 0; i < size; ++i)
          {
            auto& e = set[i];
            if (key_of(&e) != 0)
            {
              e.~Entry();
            }
          }
        }
        alloc->dealloc(set, get_size() * sizeof(Entry));
      }
    }

    // Returns true if newly added, false if previously present.
    bool insert(Alloc* alloc, Entry& entry, size_t& location)
    {
      auto orig_key = key_of(&entry);
      assert(orig_key == (size_t)get_unmarked_pointer(orig_key));

      if (size_bits == 0)
        grow(alloc);

      size_t size = get_size();
      size_t mask = size - 1;
      size_t index = verona::rt::bits::hash((void*)orig_key) & mask;
      size_t dib_entry = 0;

      for (size_t i = 0; i <= mask; i++)
      {
        auto other = &set[index];
        auto other_key = key_of(other);

        if (other_key == 0)
        {
          if (key_of(&entry) == orig_key)
            location = index;

          // This index is empty, insert here.
          set_entry(index, entry, dib_entry);

          count++;
          grow(alloc);
          return true;
        }

        size_t dib_other = get_dib(size, index, other_key);

        if (dib_entry == dib_other)
        {
          // This entry is already present. This should only happen for the
          // original o, not for any swapped pointer.
          if (
            (key_of(&entry) == orig_key) &&
            (key_of(&entry) == (size_t)get_unmarked_pointer(other_key)))
          {
            location = index;
            return false;
          }
        }
        else if (dib_entry > dib_other)
        {
          auto tmp = std::move(*other);

          if (key_of(&entry) == orig_key)
            location = index;

          // The DIB of the entry to insert is greater than the DIB of the
          // entry at this index. Insert o here, and continue looking for
          // somewhere to insert other.
          set_entry(index, entry, dib_entry);

          key_of(&tmp) = (size_t)get_pointer(key_of(&tmp));
          entry = std::move(tmp);
          dib_entry = dib_other;
        }

        // Advance both the index, wrapping around, and the DIB.
        index = (index + 1) & mask;
        dib_entry++;
      }

      assert(0);
      return false;
    }

    void insert_unique(Alloc* alloc, Entry& entry)
    {
      size_t dummy;
      auto unique = insert(alloc, entry, dummy);
      assert(unique);
      UNUSED(unique);
    }

    void sweep_set(Alloc* alloc, size_t marked)
    {
      if (size_bits == 0)
        return;

      size_t size = get_size();

      // If our marked object count is low, build a new set instead.
      if (size_bits > INITIAL_SIZE_BITS)
      {
        size_t shrink_threshold = size >> 3;

        if (marked <= shrink_threshold)
        {
          // Pick a size that can hold twice the marked count.
          shrink(alloc, marked);
          return;
        }
      }

      size_t empty_dib = 0;
      size_t fill_dib = 0;
      count = marked;

      for (size_t index = 0; index < size; index++)
      {
        auto entry = &set[index];
        auto& key = key_of(entry);

        if (key == 0)
        {
          if (fill_dib > 0)
          {
            empty_dib++;
          }
          else
          {
            empty_dib = 1;
            fill_dib = 0;
          }
        }
        else if ((key & MARK) != 0)
        {
          key = (size_t)get_unmarked_pointer(key);
          size_t dib = get_dib(size, index, key);

          if (dib == 0)
          {
            set[index] = std::move(*entry);
            empty_dib = 0;
            fill_dib = 0;
          }
          else if (empty_dib == 0)
          {
            set_entry(index, *entry, dib);
            empty_dib = 0;
            fill_dib = 0;
          }
          else
          {
            set_entry(
              index - empty_dib + fill_dib, *entry, dib - empty_dib + fill_dib);

            key_of(&set[index]) = 0;
            empty_dib++;
            fill_dib++;
          }
        }
        else
        {
          set[index].~Entry();

          key_of(&set[index]) = 0;

          if (fill_dib > 0)
          {
            empty_dib++;
          }
          else
          {
            empty_dib = 1;
            fill_dib = 0;
          }
        }
      }

      if (empty_dib > 0)
      {
        size_t mask = size - 1;

        for (size_t index = 0; index < size; index++)
        {
          auto entry = &set[index];
          auto& key = key_of(entry);

          size_t dib = get_dib(size, index, key);

          if (dib > 0)
          {
            key = (size_t)get_unmarked_pointer(key);

            set_entry(
              (index - empty_dib + fill_dib + size) & mask,
              *entry,
              dib - empty_dib + fill_dib);

            key_of(&set[index]) = 0;
            empty_dib++;
            fill_dib++;
          }
          else
          {
            break;
          }
        }
      }
    }

    void clear(Alloc* alloc)
    {
      sweep_set(alloc, 0);
    }

    Iterator find(size_t orig_key)
    {
      if (count == 0)
      {
        return end();
      }

      assert(orig_key == (size_t)get_unmarked_pointer(orig_key));

      size_t size = get_size();
      size_t mask = size - 1;
      size_t index = verona::rt::bits::hash((void*)orig_key) & mask;
      size_t dib_entry = 0;

      for (size_t i = 0; i <= mask; i++)
      {
        auto other = &set[index];
        auto key = key_of(other);

        if (key == 0)
        {
          return end();
        }

        size_t dib_other = get_dib(size, index, key);

        if (dib_entry == dib_other)
        {
          // This entry is already present. This should only happen for the
          // original o, not for any swapped pointer.
          if ((key == orig_key) && (key == (size_t)get_unmarked_pointer(key)))
          {
            return {this, index};
          }
        }
        else if (dib_entry > dib_other)
        {
          return end();
        }

        // Advance both the index, wrapping around, and the DIB.
        index = (index + 1) & mask;
        dib_entry++;
      }

      assert(0);
      return end();
    }

    void erase(void* p)
    {
      if (count == 0)
      {
        return;
      }
      auto i = find((size_t)p);
      if (i == end())
      {
        return;
      }

      auto size = get_size();
      auto mask = size - 1;
      auto cur_index = i.get_index();
      i->~Entry();
      auto next_index = (cur_index + 1) & mask;

      size_t key = 0;
      size_t dib = 0;
      while ((key = key_of(&set[next_index])) != 0 &&
             (dib = get_dib(size, next_index, key)) != 0)
      {
        set_entry(cur_index, set[next_index], dib - 1);

        cur_index = next_index;
        next_index = (next_index + 1) & mask;
      }

      key_of(&set[cur_index]) = 0;
      count--;
    }
  };
} // namespace verona::rt
