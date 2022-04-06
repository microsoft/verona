// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../ds/bag.h"
#include "../ds/stack.h"
#include "../test/logging.h"
#include "../test/systematic.h"

#include <snmalloc/snmalloc.h>

namespace verona::rt
{
  static void yield();
  /**!object.md
   * Object representation
   * =====================
   *
   * The object has at least two pointer sized fields
   *
   *  - Region meta data
   *  - Descriptor
   *
   * However, both of these have numerous meanings and bit borrowing.
   *
   * Region Meta-data
   * ----------------
   *
   * This is split into the bottom 3 bits represent the type of status, and the
   * remaining bits are the payload.
   *
   *  Tag      |  Payload                                     | Object type
   * ---------------------------------------------------------|--------------
   *  Isolate  |  Aligned pointer to region metadata object   | Root of region
   *  Unmarked |  Next pointer in region or for work list     | Mutable object
   *  Marked   |  Next pointer in region                      | Mutable object
   *  Immutable|  Reference count                             | Immutable object
   *  SCC_PTR  |  Union-find parent pointer for SCC           | Immutable object
   *  Pending  |  Depth of longest chain in SCC               | Immutable object
   *  Cown     |  Reference count                             | Cown object
   *  Open ISO |  Region specific scratch space               | Root of region
   *
   *
   * *Pending* is used during operations to construct an SCC.
   * During construction it represents the depth of an SCC, so we can produce
   * balanced SCCs.
   *
   * *Unmarked* is used primarily to mean a mutable object that has not been
   * marked by the current trace phase. But can also be used as a worklist
   * during certain operations using the `LinkedObjectStack`.
   *
   * `SCC_PTR` is used to point at an object that knows about the reference
   * count of this SCC in the immutable graph.  It could also point at Pending
   * during construction of an SCC.
   *
   * Descriptor
   * ----------
   *
   * The descriptor primarily points to a descriptor (vtable and meta data)
   * about this object. However, we again borrow the bottom bits.
   *
   * For an immutable object or a cown object, we use the bottom bits to
   * represent the scanning status of the global cown leak detector. It is
   * effectively a two state scan with A and B epoch, so we do not need to clear
   * epoch flags after scanning.
   *
   * There are additional states to represent:
   *
   * - `SCHEDULED_FOR_SCAN`: a message has been sent to this Cown to wake it up
   * and thus perform a scan of its state and message queue.
   * - `SCANNED`: this Cown has scanned its current state, but has not scanned
   * its message queue, so still needs to be rescheduled once all the messages
   * are guaranteed to be scanned before sending.
   * - `EPOCH_A`/`B`: this Cown guarantees all its state and messages have been
   * scanned, and all directly reachable Cowns have been rescheduled, so they
   * will be scanned.
   */
  using Alloc = snmalloc::Alloc;
  using namespace snmalloc;
  class Object;
  class RegionBase;

  using RefCounts = Bag<Object, uintptr_t, Alloc>;
  using RefCount = RefCounts::Elem;

  using ObjectStack = Stack<Object, Alloc>;
  static constexpr size_t descriptor_alignment =
    snmalloc::bits::min<size_t>(8, alignof(void*));

  struct alignas(descriptor_alignment) Descriptor
  {
    // for field in o do
    //  st.push(o.field)
    using TraceFunction = void (*)(const Object* o, ObjectStack& st);

    using NotifiedFunction = void (*)(Object* o);

    // We distinguish between an object's finaliser and its destructor.
    //
    // The finaliser is run while all objects in the region are valid. It may
    // follow pointers and examine other objects. On the other hand it must
    // leave the object in a valid state as well, for other finalisers to run.
    //
    // When the destructor runs, other objects of the region may have already
    // been destroyed and deallocated. Therefore the destructor must not follow
    // any of its fields. On the other hand, it may leave the object in an
    // invalid state ie. close file descriptors or deallocate some auxiliary
    // storage.
    //
    // The finaliser can must add all the subregions reachable from this object
    // to the ObjectStack it is passed, so that they can be deallocated once
    // this region has been deallocated.
    using FinalFunction = void (*)(Object* o, Object* region, ObjectStack& st);

    using DestructorFunction = void (*)(Object* o);

    size_t size;
    TraceFunction trace;
    FinalFunction finaliser;
    NotifiedFunction notified = nullptr;
    DestructorFunction destructor = nullptr;
    // TODO: virtual dispatch, pattern matching on type, reflection
  };

  enum class EpochMark : uint8_t
  {
    EPOCH_NONE = 0x0,
    EPOCH_A = 0x1,
    EPOCH_B = 0x2,
    SCHEDULED_FOR_SCAN = 0x3,
    SCANNED = 0x4,
  };

  enum class RcColour : uint8_t
  {
    GREEN = 0x0,
    RED = 0x1,
    BLACK = 0x2,
  };

  inline std::ostream& operator<<(std::ostream& os, EpochMark e)
  {
    switch (e)
    {
      case EpochMark::EPOCH_NONE:
        return os << "EPOCH_NONE";
      case EpochMark::EPOCH_A:
        return os << "EPOCH_A";
      case EpochMark::EPOCH_B:
        return os << "EPOCH_B";
      case EpochMark::SCHEDULED_FOR_SCAN:
        return os << "SCHEDULED_FOR_SCAN";
      case EpochMark::SCANNED:
        return os << "SCANNED";
      default:
        abort();
    }
  }

  enum TransferOwnership
  {
    NoTransfer,
    YesTransfer
  };

  /// The C++ representation of objects has no fields. All meta-data for the
  /// object is in the `Header` struct. Object should not be allocated directly,
  /// but instead should be allocated as part of the runtime.
  /// Contains a unique ID in systematic testing.
  class Object
  {
  public:
    enum RegionMD : uint8_t
    {
      UNMARKED = 0x0,
      MARKED = 0x1,
      SCC_PTR = 0x2,
      RC = 0x3,
      ISO = 0x4,
      PENDING = 0x5,
      NONATOMIC_RC = 0x6,
      COWN = 0x7,
      OPEN_ISO = 0x8 // TODO This is a problem for 32bit platforms. We need to
                     // fix as part of major refactor of header layout.
    };

    inline friend std::ostream& operator<<(std::ostream& os, RegionMD md)
    {
      switch (md)
      {
        case RegionMD::UNMARKED:
          return os << "UNMARKED";
        case RegionMD::MARKED:
          return os << "MARKED";
        case RegionMD::SCC_PTR:
          return os << "SCC_PTR";
        case RegionMD::RC:
          return os << "RC";
        case RegionMD::ISO:
          return os << "ISO";
        case RegionMD::PENDING:
          return os << "PENDING";
        case RegionMD::NONATOMIC_RC:
          return os << "NONATOMIC_RC";
        case RegionMD::COWN:
          return os << "COWN";
        default:
          abort();
      }
    }

#ifdef USE_SYSTEMATIC_TESTING
    inline friend std::ostream& operator<<(std::ostream& os, const Object* o)
    {
      return os << o->id<false>();
    }
#endif

    // Note that while we only need 3 bits, we need to reserve enough bits
    // for the hashmap implementation. A static assert in hashmap.h should
    // enforce this.
    static constexpr uint8_t MARK_BITS = 3;
    static constexpr uintptr_t MARK_MASK = (1 << MARK_BITS) - 1;

    // snmalloc will ensure that Objects are properly aligned. However, in some
    // situations, e.g. allocating within a RegionArena, we still need to
    // ensure that pointers to objects are aligned.
    static constexpr size_t ALIGNMENT = (1 << MIN_ALLOC_BITS);

    /// This class represents the Verona object header.
    /// It is stored directly before a Verona object.
    /// Its overall size is two pointers.
    struct alignas(ALIGNMENT) Header
    {
      union
      {
        Object* next;
        std::atomic<size_t> rc;
        size_t bits;
      };

      union
      {
        std::atomic<const Descriptor*> descriptor;
        uintptr_t descriptor_bits;
      };

#ifdef USE_SYSTEMATIC_TESTING
      // Used to give objects unique identifiers for systematic testing.
      uintptr_t sys_id;
#endif
    };

  private:
    static constexpr uintptr_t MASK = ALIGNMENT - 1;
    static constexpr uint8_t SHIFT = (uint8_t)bits::next_pow2_bits_const(MASK);
    static constexpr size_t ONE_RC = 1 << SHIFT;

#ifdef USE_SYSTEMATIC_TESTING
    // Used to give objects unique identifiers for systematic testing.
    inline static std::atomic<size_t> id_source = 0;
#endif

    Header& get_header() const
    {
      return *((Header*)real_start());
    }

    // Used to track last object that was register with a region.
    // This is used to ensure that anything of Object has just
    // been region allocated.
    static void* last_alloc(void* next)
    {
      static thread_local void* last = nullptr;
      auto prev = last;
      last = next;
      return prev;
    }

  public:
    /// Returns the start of the allocation containing this object
    /// I.e. the start of the objects verona header.
    std::byte* real_start() const
    {
      return ((std::byte*)this) - sizeof(Header);
    }

    Object(const Object&) = delete;
    Object& operator=(const Object&) = delete;

    /// Should be called by the region allocator prior to initialising an
    /// object as part of the runtime.  This is used to ensure that all
    /// subclasses of rt::Object are actually part of the runtime.
    static Object* register_object(void* base, const Descriptor* desc)
    {
      Object* obj = object_start(base);
      obj->get_header().descriptor = desc;
#ifndef NDEBUG
      last_alloc(obj);
#endif

#ifdef USE_SYSTEMATIC_TESTING
      obj->get_header().sys_id =
        id_source.fetch_add(1, std::memory_order_relaxed);
#endif

      assert(debug_is_aligned(obj));
      return obj;
    }

    /// Given a pointer to the start of the header, return a pointer to the
    /// start of the object.
    static Object* object_start(void* p)
    {
      return (Object*)((char*)p + sizeof(Object::Header));
    }

    Object()
    {
      // This assertion fails of register_object was not called prior to
      // initialising this object.  This is probably due to not being
      // allocated with the Verona region allocator
      assert(last_alloc(nullptr) == this);

      // This should have already been set up.
      assert(get_descriptor() != nullptr);
    }

    inline const Descriptor* get_descriptor() const
    {
      return (
        const Descriptor*)((uintptr_t)get_header().descriptor.load(std::memory_order_relaxed) & ~MARK_MASK);
    }

#ifdef USE_SYSTEMATIC_TESTING
    inline static void reset_ids()
    {
      id_source = 0;
    }
#endif

    template<bool scramble = true>
    inline uintptr_t id() const
    {
#ifdef USE_SYSTEMATIC_TESTING
      if constexpr (scramble)
        return Systematic::get_scrambler().perm(get_header().sys_id);
      else
#  ifdef USE_FLIGHT_RECORDER
        // If the flight recorder is enabled to debug systematic
        // testing, then we cannot assume the object still exists
        // at log time, and thus this needs to be the raw pointer.
        return (uintptr_t)this;
#  else
        return get_header().sys_id;
#  endif
#else
      return (uintptr_t)this;
#endif
    }

    bool debug_is_iso()
    {
      return get_class() == RegionMD::ISO;
    }

    bool debug_is_mutable()
    {
      return get_class() == RegionMD::UNMARKED;
    }

    bool debug_is_immutable()
    {
      switch (get_class())
      {
        case RegionMD::SCC_PTR:
        case RegionMD::RC:
        case RegionMD::NONATOMIC_RC:
          return true;

        default:
          return false;
      }
    }

    bool debug_is_cown()
    {
      return get_class() == RegionMD::COWN;
    }

    bool debug_is_rc()
    {
      return get_class() == RegionMD::RC;
    }

    bool debug_test_rc(size_t test_rc)
    {
      assert(debug_is_immutable());
      Object* o = immutable();
      return o->get_header().bits ==
        ((test_rc << SHIFT) | (uint8_t)RegionMD::RC);
    }

    intptr_t debug_rc()
    {
      return (intptr_t)get_header().bits >> SHIFT;
    }

    bool is_rc_candidate()
    {
      return get_class() == RegionMD::UNMARKED ||
        get_class() == RegionMD::MARKED || get_class() == RegionMD::ISO;
    }

    Object* debug_immutable_root()
    {
      return immutable();
    }

    Object* debug_next()
    {
      assert(debug_is_iso() || debug_is_mutable());
      return (Object*)(get_header().bits & ~MASK);
    }

    static bool debug_is_aligned(const void* o)
    {
      return ((uintptr_t)o & MASK) == 0;
    }

  private:
    friend class Cown;
    friend class Immutable;
    friend class Freeze;
    friend class Region;
    friend class RegionBase;
    friend class RegionTrace;
    friend class RegionArena;
    friend class RegionRc;
    friend class RememberedSet;
    friend class ExternalReferenceTable;
    template<typename Entry>
    friend class ObjectMap;
    friend class Message;
    friend class LocalEpoch;
    friend size_t debug_get_ref_count(Object* o);

    friend class LinkedObjectStack;

    template<typename T>
    friend class Noticeboard;

  private:
    inline RegionMD get_class()
    {
      return (RegionMD)(get_header().bits & MASK);
    }

    inline size_t size()
    {
      return get_descriptor()->size;
    }

    inline bool is_type(const Descriptor* desc)
    {
      return get_descriptor() == desc;
    }

    inline void init_iso()
    {
      get_header().bits = (size_t)this | (uint8_t)RegionMD::ISO;
    }

    inline void init_next(Object* o)
    {
      get_header().next = o;
    }

    inline Object* get_next()
    {
      assert(get_class() == RegionMD::UNMARKED);
      return get_header().next;
    }

    inline Object* get_next_any_mark()
    {
      assert(
        (get_class() == RegionMD::MARKED) ||
        (get_class() == RegionMD::UNMARKED) ||
        (get_class() == RegionMD::PENDING) || (get_class() == RegionMD::ISO));

      return (Object*)(get_header().bits & ~MASK);
    }

    inline size_t get_ref_count()
    {
      assert(
        (get_class() == RegionMD::OPEN_ISO) ||
        (get_class() == RegionMD::MARKED) ||
        (get_class() == RegionMD::UNMARKED));
      return (size_t)(get_header().bits >> SHIFT);
    }

    inline void incref_rc_region()
    {
      assert(
        (get_class() == RegionMD::OPEN_ISO) ||
        (get_class() == RegionMD::MARKED) ||
        (get_class() == RegionMD::UNMARKED));
      get_header().bits += ONE_RC;
    }

    inline void decref_rc_region()
    {
      assert(
        (get_class() == RegionMD::OPEN_ISO) ||
        (get_class() == RegionMD::MARKED) ||
        (get_class() == RegionMD::UNMARKED));
      get_header().bits -= ONE_RC;
    }

    inline void init_ref_count()
    {
      get_header().bits = RegionMD::UNMARKED + ONE_RC;
    }

    inline void init_iso_ref_count(size_t count)
    {
      assert(get_class() == RegionMD::ISO);
      get_header().bits = (count << SHIFT) | (uint8_t)RegionMD::OPEN_ISO;
    }

    inline void set_next(Object* o)
    {
      assert(get_class() == RegionMD::UNMARKED);
      get_header().next = o;
    }

  public:
    inline RegionBase* get_region()
    {
      assert(get_class() == RegionMD::ISO);
      return (RegionBase*)(get_header().bits & ~MASK);
    }

  private:
    inline void set_region(RegionBase* region)
    {
      assert(get_class() == RegionMD::ISO || get_class() == RegionMD::OPEN_ISO);
      get_header().bits = (size_t)region | (uint8_t)RegionMD::ISO;
    }

    inline Object* get_scc()
    {
      assert(get_class() == RegionMD::SCC_PTR);
      return (Object*)(get_header().bits & ~MASK);
    }

    inline void set_scc(Object* o)
    {
      get_header().bits = (size_t)o | (uint8_t)RegionMD::SCC_PTR;
    }

    inline void make_scc()
    {
      get_header().bits = (size_t)RegionMD::RC + ONE_RC;
    }

    inline void make_nonatomic_scc()
    {
      get_header().bits = (size_t)RegionMD::NONATOMIC_RC + ONE_RC;
    }

    inline void make_atomic()
    {
      assert(get_class() == RegionMD::NONATOMIC_RC);
      get_header().bits = (get_header().bits & ~MASK) | (uint8_t)RegionMD::RC;
    }

    inline void make_cown()
    {
      get_header().bits = (size_t)RegionMD::COWN + ONE_RC;
    }

    inline bool is_pending()
    {
      return get_class() == RegionMD::PENDING;
    }

    inline void set_pending_rank(size_t rank)
    {
      // If this is balanced it should never get above 64.
      assert(rank < 128);
      get_header().bits = (rank << SHIFT) | (size_t)RegionMD::PENDING;
    }

    inline void set_pending()
    {
      set_pending_rank(0);
    }

    inline size_t pending_rank()
    {
      assert(is_pending());
      return get_header().bits >> SHIFT;
    }

    inline Object* root_and_class(RegionMD& c)
    {
      c = get_class();

      switch (c)
      {
        case RegionMD::SCC_PTR:
        {
          auto parent = get_scc();
          auto curr = this;

          while ((c = parent->get_class()) == RegionMD::SCC_PTR)
          {
            auto grand_parent = parent->get_scc();
            curr->set_scc(grand_parent);
            curr = parent;
            parent = grand_parent;
          }

          assert(
            c == RegionMD::RC || c == RegionMD::NONATOMIC_RC ||
            c == RegionMD::PENDING || c == RegionMD::UNMARKED);

          return parent;
        }

        default:
        {
          return this;
        }
      }
    }

    inline Object* immutable()
    {
      RegionMD c;
      Object* r = root_and_class(c);

      assert(c == RegionMD::RC || c == RegionMD::NONATOMIC_RC);

      return r;
    }

    inline void mark()
    {
      assert(get_class() == RegionMD::UNMARKED);
      get_header().bits |= (uint8_t)RegionMD::MARKED;
    }

    inline void mark_iso()
    {
      assert(get_class() == RegionMD::ISO);
      get_header().bits |= (uint8_t)RegionMD::MARKED;
    }

    inline void unmark()
    {
      assert(get_class() == RegionMD::MARKED);
      get_header().bits &= ~(size_t)RegionMD::MARKED;
    }

    inline bool is_opened()
    {
      return get_class() == RegionMD::OPEN_ISO;
    }

  public:
    inline EpochMark get_epoch_mark()
    {
      return (EpochMark)((uintptr_t)get_header().descriptor.load() & MARK_MASK);
    }

  private:
    inline bool in_epoch(EpochMark e)
    {
      assert(
        (get_class() == RegionMD::RC) || (get_class() == RegionMD::SCC_PTR) ||
        (get_class() == RegionMD::COWN));

      return get_epoch_mark() == e;
    }

    inline void set_epoch_mark(EpochMark e)
    {
      Logging::cout() << "Object epoch: " << this << " (" << get_class() << ") "
                      << get_epoch_mark() << " -> " << e << Logging::endl;

      // We only require relaxed consistency here as we can perfectly see old
      // values as we know that we will only need up-to-date values once we have
      // completed the consensus protocol to enter the sweep phase of the LD.
      get_header().descriptor.store(
        (const Descriptor*)((uintptr_t)get_descriptor() | (size_t)e),
        std::memory_order_relaxed);
    }

    inline void set_epoch(EpochMark e)
    {
      assert(
        (get_class() == RegionMD::RC) || (get_class() == RegionMD::SCC_PTR) ||
        (get_class() == RegionMD::COWN));

      assert(
        (e == EpochMark::EPOCH_NONE) || (e == EpochMark::EPOCH_A) ||
        (e == EpochMark::EPOCH_B) || (e == EpochMark::SCANNED));

      set_epoch_mark(e);
    }

    inline void set_rc_colour(RcColour colour)
    {
      get_header().descriptor_bits =
        (get_header().descriptor_bits & ~MARK_MASK) | (uintptr_t)colour;
    }

    inline RcColour get_rc_colour()
    {
      return (RcColour)((uintptr_t)get_header().descriptor_bits & MARK_MASK);
    }

    inline bool has_ext_ref()
    {
      assert(!debug_is_immutable());

      return ((uintptr_t)get_header().descriptor.load() & (uintptr_t)1) ==
        (uintptr_t)1;
    }

    inline void set_has_ext_ref()
    {
      assert(!debug_is_immutable());
      assert(((uintptr_t)get_header().descriptor.load() & MARK_MASK) == 0);

      get_header().descriptor.store(
        (const Descriptor*)((uintptr_t)get_header().descriptor.load() | (uintptr_t)1),
        std::memory_order_relaxed);
    }

    inline void clear_has_ext_ref()
    {
      assert(!debug_is_immutable());
      get_header().descriptor.store(
        (const Descriptor*)((uintptr_t)get_header().descriptor.load() & ~(uintptr_t)1),
        std::memory_order_relaxed);
    }

    inline bool cown_marked_for_scan(EpochMark e)
    {
      assert(get_class() == RegionMD::COWN);
      EpochMark t = get_epoch_mark();
      return (t == e) || (t > EpochMark::EPOCH_B);
    }

    inline bool cown_scanned(EpochMark e)
    {
      assert(get_class() == RegionMD::COWN);
      EpochMark t = get_epoch_mark();
      return (t == e) || (t == EpochMark::SCANNED);
    }

    inline void cown_mark_for_scan()
    {
      assert(get_class() == RegionMD::COWN);
      set_epoch_mark(EpochMark::SCHEDULED_FOR_SCAN);
    }

    inline void cown_mark_scanned()
    {
      assert(get_class() == RegionMD::COWN);
      set_epoch_mark(EpochMark::SCANNED);
    }

    inline void incref_nonatomic()
    {
      assert(get_class() == RegionMD::NONATOMIC_RC);
      get_header().bits += ONE_RC;
    }

    // Returns true if you are incrementing from zero.
    inline bool incref()
    {
      assert((get_class() == RegionMD::RC) || (get_class() == RegionMD::COWN));

      return get_header().rc.fetch_add(ONE_RC) == get_class();
    }

    inline bool decref()
    {
      // This does not perform the atomic subtraction if rc == 1 on entry.
      // Otherwise, will perform the atomic subtraction, which may be the
      // last one given other concurrent decrefs.
      assert(get_class() == RegionMD::RC || get_class() == RegionMD::COWN);

      size_t done_rc = (size_t)get_class() + ONE_RC;

      size_t approx_rc = get_header().bits;
      assert(approx_rc >= ONE_RC);

      if (approx_rc != done_rc)
      {
        approx_rc = get_header().rc.fetch_sub(ONE_RC);

        if (approx_rc != done_rc)
          return false;
      }

      assert(approx_rc == done_rc);
      return true;
    }

    /**
     * Larger reference count than is possible to indicate that the cown's
     * reference count can no longer have new strong references taken out.
     **/
    static constexpr size_t FINISHED_RC =
      (((size_t)1) << (((sizeof(size_t)) * 8) - 1)) + (size_t)RegionMD::COWN;

    /**
     * Returns true, if this was the last decref on the cown.  If this returns
     * true all future, and parallel, calls to incref_cown_from_weak will return
     * false.
     **/
    inline bool decref_cown()
    {
      // This always performs the atomic subtraction, since the cown should
      // see its own rc as zero this is due to how weak reference to cowns
      // interact.  An attempt to acquire a weak reference will increase the
      // strong count.
      // The top bit of the strong count is set to indicate that the strong
      // count has reached zero, and future weak count increase should fail.
      assert(debug_rc() != 0);
      assert(get_header().rc < FINISHED_RC);
      assert(get_class() == RegionMD::COWN);
      static constexpr size_t DONE_RC = (size_t)RegionMD::COWN + ONE_RC;

      size_t prev_rc = get_header().rc.fetch_sub(ONE_RC);

      if (prev_rc != DONE_RC)
        return false;

      yield();

      size_t zero_rc = (size_t)RegionMD::COWN;
      return get_header().rc.compare_exchange_strong(zero_rc, FINISHED_RC);
    }

    /**
     * Returns true, if a strong reference was created.
     **/
    inline bool acquire_strong_from_weak()
    {
      // Check if top bit is set, if not then we have validily created a new
      // strong reference
      if (get_header().rc.fetch_add(ONE_RC) < FINISHED_RC)
        return true;

      yield();

      // We failed to create a strong reference reset rc.
      // Note store is fine, as only other operations on this will
      // be failed weak reference promotions.
      get_header().rc.store(FINISHED_RC, std::memory_order_relaxed);
      return false;
    }

    inline bool has_finaliser()
    {
      return get_descriptor()->finaliser != nullptr;
    }

    inline bool has_notified()
    {
      return get_descriptor()->notified != nullptr;
    }

    inline bool has_destructor()
    {
      return get_descriptor()->destructor != nullptr;
    }

    static inline bool is_trivial(const Descriptor* desc)
    {
      return desc->destructor == nullptr && desc->finaliser == nullptr;
    }

    inline bool is_trivial()
    {
      return is_trivial(get_descriptor());
    }

  public:
    inline bool cown_zero_rc()
    {
      assert(get_class() == RegionMD::COWN);

      return get_header().rc.load(std::memory_order_relaxed) == FINISHED_RC;
    }

  private:
    inline void trace(ObjectStack& f) const
    {
      get_descriptor()->trace(this, f);
    }

    inline void finalise(Object* region, ObjectStack& isos)
    {
      if (has_finaliser())
        get_descriptor()->finaliser(this, region, isos);
    }

    inline void notified()
    {
      if (has_notified())
        get_descriptor()->notified(this);
    }

    inline void destructor()
    {
      if (has_destructor())
        get_descriptor()->destructor(this);
    }

    inline void dealloc(Alloc& alloc)
    {
      alloc.dealloc(&this->get_header(), size());
    }

  protected:
    static void
    add_sub_region(Object* obj, Object* region, ObjectStack& sub_regions)
    {
      // Should be the entry-point of the region.
      assert(
        (region == nullptr) || (region->get_class() == Object::RegionMD::ISO) ||
        (region->get_class() == Object::RegionMD::OPEN_ISO));
      // Have to be careful about internal references to the entry point for the
      // `region` i.e. when obj == region we are refering to the entry point
      // from inside the region and should not treat this as a subregion
      // pointer.
      if ((obj != nullptr) && (obj != region))
      {
        if (obj->get_class() == Object::RegionMD::ISO)
        {
          sub_regions.push(obj);
        }
      }
    }
  };

  /// Returns the size required for a Verona object to embed the
  /// C++ object T.
  template<class T>
  static constexpr size_t vsizeof = snmalloc::bits::align_up(
    sizeof(T) + sizeof(Object::Header), Object::ALIGNMENT);

} // namespace verona::rt
