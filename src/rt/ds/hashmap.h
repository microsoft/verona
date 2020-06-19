// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "../object/object.h"

namespace verona::rt
{
  template<typename Entry>
  struct DefaultMapCallbacks
  {
    static void on_insert(Alloc*, Entry&) {}
    static void on_erase(Alloc*, Entry&) {}
  };

  /**
   * Robin Hood hash map where the key type is `K*`, where `K` is derrived from
   * `Object`. The `Entry` type must be either `K*` or `std::pair<K*, Value>`.
   *
   * Static callbacks may be supplied as the second template parameter as
   * follows:
   *   ```cpp
   *   struct MyMapCallbacks
   *   {
   *     static void on_insert(Entry& e)
   *     {
   *       // called every time an entry is inserted into the map
   *       // i.e. when `insert` returns true.
   *     }
   *
   *     static void on_erase(Entry& e)
   *     {
   *       // called every time an entry is deleted from the map
   *     }
   *   };
   *
   *   using MyMap = PtrKeyHashMap<std::pair<Object*, uint8_t>, MyMapCallbacks>;
   *   ```
   */
  template<typename Entry, typename CB = DefaultMapCallbacks<Entry>>
  class ObjectMap
  {
    Entry* slots;
    size_t filled_slots = 0;
    uint8_t capacity_shift;
    uint8_t longest_probe = 0;

    /**
     * The key type must be derrived from `Object` because the low bits are used
     * to encode a mark bit and the probe length of the entry from its ideal
     * slot.
     */
    static constexpr uintptr_t MARK_MASK = Object::ALIGNMENT >> 1;
    static constexpr uintptr_t PROBE_MASK = MARK_MASK - 1;

    static_assert((MARK_MASK & PROBE_MASK) == 0);
    static_assert(((MARK_MASK | PROBE_MASK) & ~Object::MASK) == 0);

    template<typename>
    struct inspect_entry_type : std::false_type
    {};
    template<typename K>
    struct inspect_entry_type<K*> : std::true_type
    {
      static_assert(std::is_base_of_v<Object, K>);
      using value_type = K*;
      using entry_view = value_type;
      static constexpr bool is_set = true;
    };
    template<typename K, typename V>
    struct inspect_entry_type<std::pair<K*, V>> : std::true_type
    {
      static_assert(std::is_base_of_v<Object, K>);
      using value_type = V;
      using entry_view = std::pair<K*, V*>;
      static constexpr bool is_set = false;
    };

    static_assert(
      inspect_entry_type<Entry>(),
      "Map Entry must be Object* or std::pair<Object*, V>");

    using ValueType = typename inspect_entry_type<Entry>::value_type;
    using EntryView = typename inspect_entry_type<Entry>::entry_view;
    static constexpr bool is_set = inspect_entry_type<Entry>::is_set;

    /**
     * Return a reference to the entry key.
     */
    static uintptr_t& key_of(Entry& entry)
    {
      if constexpr (is_set)
        return (uintptr_t&)entry;
      else
        return (uintptr_t&)entry.first;
    }

    /**
     * Return the original key value, where the low bits have been cleared.
     */
    static uintptr_t unmark_key(uintptr_t key)
    {
      return key & ~Object::MASK;
    }

    /**
     * Return the probe index of the entry.
     */
    static size_t probe_index(uintptr_t key)
    {
      return (size_t)(key & PROBE_MASK);
    }

    /**
     * Increase the allocation size by approximately 50%, rounded up to the
     * nearest allocation size. The entries in the previous allocation will be
     * reinserted.
     */
    void resize(Alloc* alloc)
    {
      auto prev = *this;

      const auto alloc_size = bits::next_pow2(capacity() * sizeof(Entry) * 2);
      slots = (Entry*)alloc->alloc<YesZero>(alloc_size);
      capacity_shift = bits::ctz(alloc_size / sizeof(Entry));
      filled_slots = 0;
      longest_probe = 0;

      for (auto it = prev.begin(); it != prev.end(); ++it)
      {
        if constexpr (is_set)
          insert(alloc, it.key());
        else
          insert(alloc, std::make_pair(it.key(), std::move(it.value())));
      }
    }

    /**
     * Place an entry into the map at the given index, overwriting any existing
     * entry. The probe bits of the key are set to `probe_len` and the
     * `longest_probe` value is updated if `probe_len` is greater.
     */
    template<typename E>
    void place_entry(E entry, size_t index, uint8_t probe_len)
    {
      slots[index] = std::forward<E>(entry);
      auto& key = key_of(slots[index]);
      assert(probe_len <= PROBE_MASK);
      key = (key & ~PROBE_MASK) | probe_len;
      if (probe_len > longest_probe)
        longest_probe = probe_len;
    }

    /**
     * Call a function using an `Entry&` where the key is temporarily reset to
     * the original value.
     */
    template<typename F>
    void call_cb(F f, Alloc* alloc, Entry& entry)
    {
      const auto key_swap = key_of(entry);
      key_of(entry) = unmark_key(key_of(entry));
      f(alloc, entry);
      key_of(entry) = key_swap;
    }

  public:
    /**
     * Iterator over the entries in an `ObjectMap`, starting from a slot index.
     */
    class Iterator
    {
      template<typename _Entry, typename _CB>
      friend class ObjectMap;

      const ObjectMap& map;
      size_t index;

      Entry& entry()
      {
        return map.slots[index];
      }

      Iterator(const ObjectMap& m, size_t i) : map(m), index(i) {}

    public:
      Object* key()
      {
        return (Object*)unmark_key(key_of(entry()));
      }

      template<bool v = !is_set, typename = typename std::enable_if_t<v>>
      ValueType& value()
      {
        return entry().second;
      }

      bool is_marked()
      {
        return key_of(entry()) & MARK_MASK;
      }

      void mark()
      {
        key_of(entry()) |= MARK_MASK;
      }

      EntryView operator*()
      {
        if constexpr (is_set)
          return key();
        else
          return std::make_pair(key(), &value());
      }

      Iterator& operator++()
      {
        while (++index < map.capacity())
        {
          const auto key = key_of(map.slots[index]);
          if (key != 0)
            break;
        }
        return *this;
      }

      bool operator==(const Iterator& other) const
      {
        return (index == other.index) && (&map == &other.map);
      }

      bool operator!=(const Iterator& other) const
      {
        return !(*this == other);
      }
    };

    /**
     * Create an `ObjectMap` with an initial capacity for at least 8 entries.
     */
    ObjectMap(Alloc* alloc)
    {
      static constexpr size_t init_alloc =
        bits::next_pow2_const(sizeof(Entry) * 8);
      capacity_shift = bits::ctz(init_alloc / sizeof(Entry));
      slots = (Entry*)alloc->alloc<init_alloc, YesZero>();
    }

    ~ObjectMap()
    {
      dealloc(ThreadAlloc::get());
    }

    static ObjectMap<Entry, CB>* create(Alloc* alloc)
    {
      return new (alloc->alloc<sizeof(ObjectMap<Entry, CB>)>())
        ObjectMap(alloc);
    }

    void dealloc(Alloc* alloc)
    {
      clear(alloc);
      alloc->dealloc(slots, capacity() * sizeof(Entry));
    }

    /**
     * Return the amount of entries in this map.
     */
    size_t size() const
    {
      return filled_slots;
    }

    /**
     * Return the capacity for entries in the map. Note that this should not be
     * used to approximate when the map will resize.
     */
    size_t capacity() const
    {
      return (1 << capacity_shift);
    }

    Iterator begin() const
    {
      auto it = Iterator(*this, 0);
      if (unmark_key(key_of(slots[0])) == 0)
        ++it;

      return it;
    }

    Iterator end() const
    {
      return Iterator(*this, capacity());
    }

    /**
     * Find an entry in the map with the given key and return an iterator to the
     * corresponding entry. If no entry exitsts, the return value will be equal
     * to the return value of `end()`.
     */
    Iterator find(const Object* key) const
    {
      if (key == nullptr)
        return end();

      const auto hash = bits::hash((void*)key);
      auto index = hash % capacity();
      for (size_t probe_len = 0; probe_len <= longest_probe; probe_len++)
      {
        const auto k = unmark_key(key_of(slots[index]));
        if (k == (uintptr_t)key)
          return Iterator(*this, index);

        if (++index == capacity())
          index = 0;
      }

      return end();
    }

    /**
     * Insert an entry into the map. The first element of the returned pair will
     * be true if a new key is inserted, and false if an existing entry is
     * updated. The second element of the returned pair is an iterator to the
     * inserted entry. The key of the inserted entry must not be null.
     */
    template<typename E>
    std::pair<bool, Iterator> insert(Alloc* alloc, E entry)
    {
      // Resize at ~.8 load factor.
      if (size() >= ((capacity() * 4) / 5))
        resize(alloc);

      assert(key_of(entry) != 0);
      const auto key = unmark_key(key_of(entry));
      const auto hash = bits::hash((void*)key);
      auto index = hash % capacity();

      for (uint8_t probe_len = 0; probe_len <= PROBE_MASK; probe_len++)
      {
        const auto k = key_of(slots[index]);

        if (unmark_key(k) == key)
        { // Update existing entry.
          if constexpr (!is_set)
            entry.second = std::forward<E>(entry).second;

          return std::make_pair(false, Iterator(*this, index));
        }

        if (k == 0)
        { // Place into empty slot.
          place_entry(std::forward<E>(entry), index, probe_len);
          assert(!(key_of(slots[index]) & MARK_MASK));
          filled_slots++;
          call_cb(CB::on_insert, alloc, slots[index]);
          return std::make_pair(true, Iterator(*this, index));
        }

        if (probe_index(k) < probe_len)
        { // Robin Hood time. Swap with current slot and continue.
          Entry swap = std::move(slots[index]);
          place_entry(std::forward<E>(entry), index, probe_len);
          entry = swap;
          probe_len = probe_index(key_of(entry));
        }

        if (++index == capacity())
          index = 0;
      }

      // Maximum probe length reached, resize and retry.
      resize(alloc);
      return insert(alloc, std::forward<E>(entry));
    }

    /**
     * Remove an entry from the map corresponding to the given key. The return
     * value is false if no entry was found for the key and true otherwise.
     */
    bool erase(Alloc* alloc, const Object* key)
    {
      auto it = find(key);
      if (it == end())
        return false;

      erase(alloc, it);
      return true;
    }

    /**
     * Remove an entry from the map at the given iterator position. The iterator
     * must be valid. This operation will not invalidate the iterator.
     */
    void erase(Alloc* alloc, Iterator& it)
    {
      assert(key_of(it.entry()) != 0);

      // No tombstones are necessary because our insertion algorithm relies on
      // the the minimum probe length determined by the maximum value we can
      // stash in the lower bits of the key.

      call_cb(CB::on_erase, alloc, it.entry());
      it.entry().~Entry();
      key_of(it.entry()) = 0;
      filled_slots--;
    }

    /**
     * Empty the map, removing all entries. If `callback` is false, then the
     * entries removed will not have the callback `on_erase` called with the
     * removed entries.
     */
    void clear(Alloc* alloc, bool callback = true)
    {
      for (auto it = begin(); it != end(); ++it)
      {
        if (callback)
          call_cb(CB::on_erase, alloc, it.entry());

        it.entry().~Entry();
        key_of(it.entry()) = 0;
      }
      longest_probe = 0;
      filled_slots = 0;
    }

    /**
     * Erase all unmarked entries from the map and unmark the remaining entries.
     */
    void sweep(Alloc* alloc)
    {
      for (auto it = begin(); it != end(); ++it)
      {
        if (!it.is_marked())
          erase(alloc, it);
        else
          key_of(it.entry()) &= ~MARK_MASK;
      }
    }

    /**
     * Return a string representation of the map showing empty slots (`∅`),
     * key positions, and probe lengths.
     */
    template<typename OutStream>
    OutStream& debug_layout(OutStream& out) const
    {
      out << "{";
      for (size_t i = 0; i < capacity(); i++)
      {
        const auto key = key_of(slots[i]);
        if (key == 0)
        {
          out << " ∅";
          continue;
        }
        out << " (" << (const Object*)unmark_key(key_of(slots[i])) << ", probe "
            << (size_t)probe_index(key) << ")";
      }
      out << " } cap: " << capacity();
      return out;
    }
  };
}
