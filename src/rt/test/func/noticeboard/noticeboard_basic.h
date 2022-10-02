// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

/**
 * This test aims to cause a noticeboard peek to occur
 * at the same time as an update. This is testing to check
 * that the Epoch mechanism correctly protects the two steps
 * of noticeboard peek:
 *    1. Read the current value
 *    2. Incref the value read
 * An interleaving between 1 and 2 that deallocates the read value
 * would be problematic.
 *
 * This test should fail if the Epoch protection is removed from
 *   Noticeboard::peek
 * This is true for the commit that adds this comment.
 */

#include <test/harness.h>

namespace noticeboard_basic
{
  struct Alive : public VCown<Alive>
  {
  public:
    int n = 10;
  };

  struct Ping : public VBehaviour<Ping>
  {
    void* target;

    Ping(void* t) : target(t) {}

    void f()
    {
      Logging::cout() << "Ping on " << target << std::endl;
    }
  };

  struct C : public V<C>
  {
  public:
    int x = 0;
    C* next = nullptr;
    Alive* alive = nullptr;

    C(int x_) : x(x_) {}

    void trace(ObjectStack& st) const
    {
      if (next != nullptr)
        st.push(next);
      if (alive != nullptr)
        st.push(alive);
    }
  };

  struct DB : public VCown<DB>
  {
  public:
    Noticeboard<Object*> box;
    int n_max = 40;
    int n = 0;

    DB(Object* c) : box{c}
    {
#ifdef USE_SYSTEMATIC_TESTING_WEAK_NOTICEBOARDS
      register_noticeboard(&box);
#endif
    }

    void trace(ObjectStack& fields) const
    {
      box.trace(fields);
    }
  };

  enum Phase
  {
    INIT,
    PEEK
  };

  struct Peeker : public VCown<Peeker>
  {
  public:
    DB* db;
    Noticeboard<Object*>* box;
    Alive* alive = nullptr;

    Peeker(DB* db_, Noticeboard<Object*>* box_) : db(db_), box(box_) {}

    void trace(ObjectStack& fields) const
    {
      if (alive != nullptr)
      {
        fields.push(alive);
      }
      check(db);
      fields.push(db);
    }
  };

  struct UpdateDB : public VBehaviour<UpdateDB>
  {
    DB* db;
    UpdateDB(DB* db) : db(db) {}

    void f()
    {
      auto& alloc = ThreadAlloc::get();

      C* new_c = new (RegionType::Trace) C(1);

      Logging::cout() << "Update DB Create C " << new_c << std::endl;

      freeze(new_c);
      db->box.update(alloc, new_c);

      // Try to trigger a rapid collection.
      // Disable yield to avoid being preempted.
      Systematic::disable_yield();
      for (int i = 0; i < 100024; i++)
      {
        Epoch e(alloc);
        UNUSED(e);
      }
      Systematic::enable_yield();

      Logging::cout() << "Update DB Done" << std::endl;
    }
  };

  struct ToPeek : public VBehaviour<ToPeek>
  {
    Peeker* peeker;
    ToPeek(Peeker* peeker) : peeker(peeker) {}

    void f()
    {
      auto& alloc = ThreadAlloc::get();

      Logging::cout() << "Peek" << std::endl;
      auto o = (C*)peeker->box->peek(alloc);
      Logging::cout() << "Peeked " << o << std::endl;
      // o goes out of scope
      Immutable::release(alloc, o);
    }
  };

  void run_test()
  {
    Alloc& alloc = ThreadAlloc::get();

    Alive* alive = new (alloc) Alive;
    Logging::cout() << "Alive" << alive << std::endl;

    C* c = new (RegionType::Trace) C(0);
    c->next = new (RegionType::Trace) C(10);

    RegionTrace::insert(alloc, c, alive);
    c->alive = alive;
    Logging::cout() << "Create C " << c << " with alive " << alive << std::endl;

    Logging::cout() << "Create C next" << c->next << std::endl;

    freeze(c);

    DB* db = new DB(c);
    Logging::cout() << "DB " << db << std::endl;

    Peeker* peeker = new Peeker(db, &db->box);

    Logging::cout() << "Peeker " << peeker << std::endl;

    Cown::schedule<ToPeek>(peeker, peeker);
    Cown::schedule<UpdateDB>(peeker->db, peeker->db);
    Cown::schedule<Ping>(alive, alive);

    Cown::release(alloc, alive);
    Cown::release(alloc, peeker);
    // Ownership of db was transferred to Peeker, no need to release it.
  }
}
