// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../ds/forward_list.h"
#include "../ds/mpscq.h"
#include "../region/region.h"
#include "../test/systematic.h"
#include "base_noticeboard.h"
#include "multimessage.h"
#include "priority.h"
#include "schedulerthread.h"

namespace verona::rt
{
  using namespace snmalloc;
  class Cown;
  using CownThread = SchedulerThread<Cown>;
  using Scheduler = ThreadPool<CownThread>;

  static void yield()
  {
    Scheduler::get().sync.yield(Scheduler::local());
  }

  /**
   * Return output of `Systematic::coin` when systematic testing is enabled;
   * always return false otherwise.
   **/
  static bool coin(size_t range_bits = 1)
  {
#ifdef USE_SYSTEMATIC_TESTING
    return Systematic::coin(range_bits);
#else
    UNUSED(range_bits);
    return false;
#endif
  }

  static Behaviour unmute_behaviour{Behaviour::Descriptor::empty()};

  /**
   * A cown, or concurrent owner, encapsulates a set of resources that may be
   * accessed by a single (scheduler) thread at a time. A cown can only be in
   * exactly one of the following states:
   *   1. Unscheduled
   *   2. Scheduled, in the queue of a single scheduler thread
   *   3. Running on a single scheduler thread
   *
   * Once a cown is running, it executes a batch of multi-message behaviours.
   * Each message may either acquire the running cown for participation in a
   * future behaviour, or execute the behaviour if it is the last cown to be
   * acquired. If the running cown is acquired for a future behaviour, it will
   * be descheduled until that behaviour has completed.
   */
  class Cown : public Object
  {
    using MessageBody = MultiMessage::MultiMessageBody;

  public:
    enum TryFastSend
    {
      NoTryFast,
      YesTryFast
    };

    Cown(bool initialise = true)
    {
      make_cown();

      if (initialise)
      {
        auto alloc = ThreadAlloc::get();
        auto epoch = Scheduler::alloc_epoch();
        set_epoch(epoch);
        queue.init(stub_msg(alloc));
        CownThread* local = Scheduler::local();

        if (local != nullptr)
        {
          set_owning_thread(local);
          next = local->list;
          local->list = this;
          local->total_cowns++;
        }
        else
        {
          set_owning_thread(nullptr);
          next = nullptr;
        }
      }
    }

  private:
    friend class DLList<Cown>;
    friend class MultiMessage;
    friend CownThread;

    template<typename T>
    friend class Noticeboard;

    template<typename T>
    friend class SPMCQ;

    static constexpr auto NO_EPOCH_SET = (std::numeric_limits<uint64_t>::max)();

    union
    {
      std::atomic<Cown*> next_in_queue;
      uint64_t epoch_when_popped = NO_EPOCH_SET;
    };

    // Seven pointer overhead compared to an object.
    verona::rt::MPSCQ<MultiMessage> queue;

    // Used for garbage collection of cyclic cowns only.
    // Uses the bottom bit to indicate the cown has been collected
    // If the object is collected by the leak detector, we should not
    // collect again when the weak reference count hits 0.
    std::atomic<uintptr_t> thread_status;
    Cown* next;

    /**
     * Cown's weak reference count.  This keeps the cown itself alive, but not
     * the data it can reach.  Weak reference can be promoted to strong, if a
     * strong reference still exists.
     **/
    std::atomic<size_t> weak_count = 1;

    std::atomic<BPState> bp_state{};

    static Cown* create_token_cown()
    {
      static constexpr Descriptor desc = {
        vsizeof<Cown>, nullptr, nullptr, nullptr};
      auto alloc = ThreadAlloc::get();
      auto p = alloc->alloc<desc.size>();
      auto o = Object::register_object(p, &desc);
      auto a = new (o) Cown(false);

      a->cown_mark_scanned();
      return a;
    }

    static constexpr uintptr_t collected_mask = 1;
    static constexpr uintptr_t thread_mask = ~collected_mask;

    void set_owning_thread(SchedulerThread<Cown>* owner)
    {
      thread_status = (uintptr_t)owner;
    }

    void mark_collected()
    {
      thread_status |= 1;
    }

    bool is_collected()
    {
      return (thread_status.load(std::memory_order_relaxed) & collected_mask) !=
        0;
    }

    SchedulerThread<Cown>* owning_thread()
    {
      return (
        SchedulerThread<
          Cown>*)(thread_status.load(std::memory_order_relaxed) & thread_mask);
    }

  public:
#ifdef USE_SYSTEMATIC_TESTING_WEAK_NOTICEBOARDS
    std::vector<BaseNoticeboard*> noticeboards;

    void flush_all(Alloc* alloc)
    {
      for (auto b : noticeboards)
      {
        b->flush_all(alloc);
      }
    }

    void flush_some(Alloc* alloc)
    {
      for (auto b : noticeboards)
      {
        b->flush_some(alloc);
      }
    }

    void register_noticeboard(BaseNoticeboard* nb)
    {
      noticeboards.push_back(nb);
    }

#endif

    void reschedule()
    {
      if (queue.wake())
      {
        Cown::acquire(this);
        schedule();
      }
    }

    bool can_lifo_schedule()
    {
      // TODO: correctly indicate if this cown can be lifo scheduled.
      // This requires some form of pinning.
      return false;
    }

    void wake()
    {
      queue.wake();
    }

    static void acquire(Object* o)
    {
      Systematic::cout() << "Cown " << o << " acquire" << Systematic::endl;
      assert(o->debug_is_cown());
      o->incref();
    }

    static void release(Alloc* alloc, Cown* o)
    {
      Systematic::cout() << "Cown " << o << " release" << Systematic::endl;
      assert(o->debug_is_cown());
      Cown* a = ((Cown*)o);

      // Perform decref
      bool last = o->decref_cown();
      yield();

      if (!last)
        return;

      // All paths from this point must release the weak count owned by the
      // strong count.

      Systematic::cout() << "Cown " << o << " dealloc" << Systematic::endl;

      // During teardown don't recursively delete.
      if (Scheduler::is_teardown_in_progress())
      {
        // If we call weak_release here, the object will be fully collected
        // as the thread field may have been nulled during teardown.  Just
        // remove weak count, so that we collect stub in teardown phase 2.
        a->weak_count.fetch_sub(1);
        return;
      }

      // During a sweep phase check is the target has not been marked
      // and do not recursively delete if already found unreachable.
      auto local = Scheduler::local();
      if (local != nullptr && local->in_sweep_state())
      {
        if (!o->is_live(Scheduler::epoch()))
        {
          Systematic::cout()
            << "Not performing recursive deallocation on: " << o
            << Systematic::endl;
          // The cown may have already been swept, just remove weak count, let
          // sweeping/cown stub collection deal with the rest.
          a->weak_count.fetch_sub(1);
          return;
        }
      }

      // If last, then collect the cown body.
      if (!a->is_collected())
        // Queue_collect calls weak release.
        a->queue_collect(alloc);
      else
        a->weak_release(alloc);
    }

    /**
     * Release a weak reference to this cown.
     **/
    void weak_release(Alloc* alloc)
    {
      Systematic::cout() << "Cown " << this << " weak release"
                         << Systematic::endl;
      if (weak_count.fetch_sub(1) == 1)
      {
        auto* t = owning_thread();
        yield();
        if (!t)
        {
          // Deallocate an unowned cown
          Systematic::cout()
            << "Not allocated on a Verona thread, so deallocating: " << this
            << Systematic::endl;
          assert(epoch_when_popped == NO_EPOCH_SET);
          dealloc(alloc);
          return;
        }
        // Register that the epoch should be moved on
        {
          Epoch e(alloc);
          e.add_pressure();
        }
        // Tell owning thread that it has a free cown to collect.
        t->free_cowns++;
        yield();
      }
    }

    void weak_acquire()
    {
      Systematic::cout() << "Cown " << this << " weak acquire"
                         << Systematic::endl;
      assert(weak_count > 0);
      weak_count++;
    }

    /**
     * Gets a strong reference from a weak reference.
     *
     * Weak reference is preserved.
     *
     * Returns true is strong reference created.
     **/
    bool acquire_strong_from_weak()
    {
      return Object::acquire_strong_from_weak();
    }

    static void mark_for_scan(Object* o, EpochMark epoch)
    {
      Cown* cown = (Cown*)o;

      if (cown->cown_marked_for_scan(epoch))
      {
        Systematic::cout() << "Already marked " << cown << " ("
                           << cown->get_epoch_mark() << ")" << Systematic::endl;
        return;
      }

      yield();

      // This may mark for scan something that has already been scanned, due
      // to racing over the epoch mark. This is ok.
      cown->cown_mark_for_scan();

      yield();

      cown->reschedule();
    }

    void mark_notify()
    {
      if (queue.mark_notify())
      {
        Cown::acquire(this);
        schedule();
      }
      yield();
    }

  protected:
    void schedule()
    {
      // This should only be called if the cown is known to have been
      // unscheduled, for example when detecting a previously empty message
      // queue on send, or when rescheduling after a multi-message.
      CownThread* t = Scheduler::local();

      if (t != nullptr)
      {
        t->schedule_fifo(this);
        return;
      }

      // TODO this should be checked further up the stack.
      // TODO Make this assertion pass.
      // assert(can_lifo_schedule() || Scheduler::debug_not_running());

      t = Scheduler::round_robin();
      t->schedule_lifo(this);
    }

  private:
    bool in_epoch(EpochMark epoch)
    {
      bool result = Object::in_epoch(epoch);
      yield();
      return result;
    }

    void dealloc(Alloc* alloc)
    {
      Object::dealloc(alloc);
      yield();
    }

    bool scanned(EpochMark epoch)
    {
      return in_epoch(epoch);
    }

    void scan(Alloc* alloc, EpochMark epoch)
    {
      // Scan our data for cown references.
      if (!cown_scanned(epoch))
      {
        cown_mark_scanned();

        ObjectStack f(alloc);
        trace(f);
        scan_stack(alloc, epoch, f);
      }
    }

    static void scan_stack(Alloc* alloc, EpochMark epoch, ObjectStack& f)
    {
      while (!f.empty())
      {
        Object* o = f.pop();
        switch (o->get_class())
        {
          case RegionMD::ISO:
            Systematic::cout()
              << "Object Scan: reaches region: " << o << Systematic::endl;
            Region::cown_scan(alloc, o, epoch);
            break;

          case RegionMD::RC:
          case RegionMD::SCC_PTR:
            Systematic::cout()
              << "Object Scan: reaches immutable: " << o << Systematic::endl;
            Immutable::mark_and_scan(alloc, o, epoch);
            break;

          case RegionMD::COWN:
            Systematic::cout()
              << "Object Scan: reaches cown " << o << Systematic::endl;
            Cown::mark_for_scan(o, epoch);
            break;

          default:
            abort();
        }
      }
    }

    void cown_notified()
    {
      // This is not a message make sure we know that.
      // TODO: Back pressure.  This means that a notification that sends to
      // an overloaded cown will not mute this cown.  We could set up a fake
      // message structure, or alter how the backpressure system determines
      // which is/are the currently active cowns.
      Scheduler::local()->message_body = nullptr;
      notified();
    }

    /**
     * A "synchronous" version of multi-message send, to be used by
     * Cown::run_step and Cown::schedule.
     *
     * Assumes that cowns [0, index) have already been acquired. Tries to
     * acquire the remaining cowns [index, count).
     *
     * Sends a multi-message to `cowns[index]`. If the cown can be acquired
     * immediately without rescheduling (i.e. its queue was sleeping), then we
     * send the next message to try to acquire the next cown. We repeat this
     * until:
     *
     * (1) The target cown was not sleeping (i.e. it is scheduled, running, or
     *     has already been acquired in a multi-message). This means we are done
     *     here, and have to wait for that cown to run and then handle our
     *     message.
     * (2) We sent the message to the last cown. There are no further cowns to
     *     acquire, so we schedule the last cown so it can handle the
     *     multi-message behaviour.
     *     TODO: It would be semantically valid to execute the behaviour without
     *     rescheduling. However, for fairness, it is better to reschedule in
     *     case the behaviour executes for a very long time.
     **/
    static void fast_send(MultiMessage::MultiMessageBody* body, EpochMark epoch)
    {
      auto* alloc = ThreadAlloc::get();
      const auto last = body->count - 1;
      assert(body->index <= last);

      auto high_priority = false;
      if (body->index == 0)
      {
        // If priority is needed for any cown in this message, start unmuting
        // cowns in the body so that they can start running messages in their
        // queue.
        high_priority = std::any_of(
          &body->cowns[0], &body->cowns[body->count], [](const auto* c) {
            return c->bp_state.load(std::memory_order_acquire).high_priority();
          });
      }

      for (; body->index < body->count; body->index++)
      {
        auto m = MultiMessage::make_message(alloc, body, epoch);
        auto* next = body->cowns[body->index];
        Systematic::cout() << "MultiMessage " << m << ": fast requesting "
                           << next << ", index " << body->index
                           << Systematic::endl;

        if (body->index > 0)
        {
          // Double check the priority of the most recently acquired cown to
          // prevent deadlock.
          auto* cur = body->cowns[body->index - 1];
          high_priority = high_priority ||
            cur->bp_state.load(std::memory_order_acquire).high_priority() ||
            coin(3);
          yield();
          if (!high_priority)
            high_priority = cur->set_blocker(next);
        }

        if (next->overloaded() || coin(3))
          high_priority = true;

        if (!high_priority)
        {
          if (!next->try_fast_send(m))
            return;
        }
        else
        {
          // Hold epoch in case priority needs to be raised after message is
          // placed in queue.
          Epoch e(alloc);
          if (!next->try_fast_send(m))
          {
            backpressure_unblock(next, e);
            return;
          }
        }

        Systematic::cout() << "MultiMessage " << m << ": fast acquire cown "
                           << next << Systematic::endl;
        if (body->index == last)
        {
          // Case 2: acquired the last cown.
          Systematic::cout()
            << "MultiMessage " << m
            << ": fast send complete, reschedule last cown" << Systematic::endl;
          next->schedule();
          return;
        }

        // The cown was asleep, so we have acquired it now. Dequeue the
        // message because we want to handle it now. Note that after
        // dequeueing, the queue may be non-empty: the scheduler may have
        // allowed another multi-message to request and send another message
        // to this cown. However, we are guaranteed to be the first message in
        // the queue.
        const auto* m2 = next->queue.dequeue(alloc);
        assert(m == m2);
        UNUSED(m2);
      }
    }

    /**
     * This method implements an optimized multi-message send to a cown. A
     * sleeping cown will not be reschdeuled because we want to immediately
     * acquire the cown without going through the scheduler queue. Returns true
     * if the cown was asleep and needs scheduling; returns false otherwise.
     **/
    bool try_fast_send(MultiMessage* m)
    {
#ifdef USE_SYSTEMATIC_TESTING_WEAK_NOTICEBOARDS
      flush_all(ThreadAlloc::get());
      yield();
#endif
      Systematic::cout() << "Enqueue MultiMessage " << m << Systematic::endl;
      bool needs_scheduling = queue.enqueue(m);
      yield();
      if (needs_scheduling)
      {
        Cown::acquire(this);
      }
      return needs_scheduling;
    }

    /**
     * Execute the behaviour of the given multi-message.
     *
     * If the multi-message has not completed, then we will send a message to
     * the next cown to acquire.
     *
     * Otherwise, all cowns have been acquired and we can execute the message
     * behaviour.
     **/
    static bool run_step(MultiMessage* m)
    {
      MultiMessage::MultiMessageBody& body = *(m->get_body());
      Alloc* alloc = ThreadAlloc::get();
      size_t last = body.count - 1;
      auto cown = body.cowns[m->get_body()->index];

      EpochMark e = m->get_epoch();

      Systematic::cout() << "MultiMessage " << m << " index " << body.index
                         << " acquired " << cown << " epoch " << e
                         << Systematic::endl;

      // If we are in should_scan, and we observe a message in this epoch,
      // then all future messages must have been sent while in pre-scan or
      // later. Thus any messages that weren't implicitly scanned on send,
      // will be counted as inflight
      if (Scheduler::should_scan() && e == Scheduler::local()->send_epoch)
      {
        // TODO: Investigate systematic testing coverage here.
        if (cown->get_epoch_mark() != Scheduler::local()->send_epoch)
        {
          cown->scan(alloc, Scheduler::local()->send_epoch);
          cown->set_epoch_mark(Scheduler::local()->send_epoch);
        }
      }

      if (body.index < last)
      {
        if (e != Scheduler::local()->send_epoch)
        {
          Systematic::cout()
            << "Message not in current epoch" << Systematic::endl;
          // We can only see messages from other epochs during the prescan and
          // scan phases.  The message epochs must be up-to-date in all other
          // phases.  We can also see messages sent by threads that have made
          // it into PreScan before us. But the global state must be PreScan,
          // we just haven't moved into it yet. `debug_in_prescan` accounts
          // for either the local or the global state is prescan.
          assert(Scheduler::should_scan() || Scheduler::debug_in_prescan());

          if (e != EpochMark::EPOCH_NONE)
          {
            Systematic::cout() << "Message old" << Systematic::endl;

            // Count message as this must be an old message being resent for a
            // further acquisition.
            Scheduler::record_inflight_message();
            e = EpochMark::EPOCH_NONE;
          }

          assert(e == EpochMark::EPOCH_NONE);
        }
        else if (Scheduler::should_scan())
        {
          if (cown->get_epoch_mark() != Scheduler::local()->send_epoch)
          {
            Systematic::cout()
              << "Contains unscanned cown." << Systematic::endl;

            // Count message as this contains a cown, that has a message queue
            // that could potentially have old messages in.
            Scheduler::record_inflight_message();
            e = EpochMark::EPOCH_NONE;
          }
        }

        // Try to acquire as many cowns as possible without rescheduling,
        // starting from the next cown.
        body.index++;

        fast_send(&body, e);
        return false;
      }

      if (e == EpochMark::EPOCH_NONE)
      {
        // decrement counter as it must have been incremented earlier for the
        // message send
        Scheduler::recv_inflight_message();
      }

      if (Scheduler::should_scan())
      {
        if (e != Scheduler::local()->send_epoch)
        {
          Systematic::cout() << "Trace message: " << m << Systematic::endl;

          // Scan cowns for this message, as they may not have been scanned
          // yet.
          for (size_t i = 0; i < body.count; i++)
          {
            Systematic::cout()
              << "Scanning cown " << body.cowns[i] << Systematic::endl;
            body.cowns[i]->scan(alloc, Scheduler::local()->send_epoch);
          }

          // Scan closure
          ObjectStack f(alloc);
          body.behaviour->trace(f);
          scan_stack(alloc, Scheduler::local()->send_epoch, f);
        }
        else
        {
          Systematic::cout() << "Trace message not required: " << m << " (" << e
                             << ")" << Systematic::endl;
        }
      }

      Scheduler::local()->message_body = &body;

      for (size_t i = 0; i < body.count; i++)
        body.cowns[i]->set_blocker(nullptr);

      // Run the behaviour.
      body.behaviour->f();

      for (size_t i = 0; i < body.count; i++)
        Cown::release(alloc, body.cowns[i]);

      Systematic::cout() << "MultiMessage " << m << " completed and running on "
                         << cown << Systematic::endl;

      // Free the body and the behaviour.
      alloc->dealloc(body.behaviour, body.behaviour->size());
      alloc->dealloc<sizeof(MultiMessage::MultiMessageBody)>(m->get_body());

      return true;
    }

  public:
    template<
      class Behaviour,
      TransferOwnership transfer = NoTransfer,
      typename... Args>
    static void schedule(Cown* cown, Args&&... args)
    {
      schedule<Behaviour, transfer, Args...>(
        1, &cown, std::forward<Args>(args)...);
    }

    /**
     * Sends a multi-message to the first cown we want to acquire.
     *
     * Pass `transfer = YesTransfer` as a template argument if the
     * caller is transfering ownership of a reference count on each cown to
     *this method.
     **/
    template<
      class Be,
      TransferOwnership transfer = NoTransfer,
      typename... Args>
    static void schedule(size_t count, Cown** cowns, Args&&... args)
    {
      static_assert(std::is_base_of_v<Behaviour, Be>);
      Systematic::cout() << "Schedule behaviour of type: " << typeid(Be).name()
                         << Systematic::endl;

      auto* alloc = ThreadAlloc::get();
      auto* be =
        new ((Be*)alloc->alloc<sizeof(Be)>()) Be(std::forward<Args>(args)...);
      auto** sort = (Cown**)alloc->alloc(count * sizeof(Cown*));
      memcpy(sort, cowns, count * sizeof(Cown*));

#ifdef USE_SYSTEMATIC_TESTING
      std::sort(&sort[0], &sort[count], [](Cown*& a, Cown*& b) {
        return a->id() < b->id();
      });
#else
      std::sort(&sort[0], &sort[count]);
#endif

      if constexpr (transfer == NoTransfer)
      {
        for (size_t i = 0; i < count; i++)
          Cown::acquire(sort[i]);
      }

      auto body = MultiMessage::make_body(alloc, count, sort, be);

      // TODO what if this thread is external.
      //  EPOCH_A okay as currently only sending externally, before we start
      //  and thus its okay.
      //  Need to use another value when we add pinned cowns.
      auto sched = Scheduler::local();
      auto epoch = sched == nullptr ? EpochMark::EPOCH_A : Scheduler::epoch();

      if (epoch == EpochMark::EPOCH_NONE)
      {
        Scheduler::record_inflight_message();
      }

      if ((sched != nullptr) && (sched->message_body != nullptr))
        backpressure_scan(*sched->message_body, *body);

      // Try to acquire as many cowns as possible without rescheduling,
      // starting from the beginning.
      fast_send(body, epoch);
    }

    /// Transition a cown between backpressure states. Return the previous
    /// state. An attempt to set the state to Normal may be preempted by
    /// another thread setting the cown to any state that isn't Muted. Normal
    /// priority may overwrite High priority when the exact flag is set.
    ///
    /// Transitioning cowns to High priority should be done through
    /// `backpressure_unblock`.
    inline Priority backpressure_transition(Priority state, bool exact = false)
    {
      auto bp = bp_state.load(std::memory_order_acquire);
      Priority prev;
      do
      {
        yield();
        prev = bp.priority();

        if ((state == Priority::Normal) && (prev != Priority::Low) && !exact)
          return prev;

        if (prev == state)
          return prev;

        // When testing spurious failure, simulated by the `coin` returning
        // false, the exchange must not occur.
      } while (coin(9) ||
               !bp_state.compare_exchange_weak(
                 bp,
                 BPState() | bp.blocker() | state | bp.has_token(),
                 std::memory_order_acq_rel));

      Systematic::cout() << "Cown " << this << ": backpressure state " << prev
                         << " -> " << state << Systematic::endl;
      yield();

      if (prev == Priority::Low)
      {
        auto sleeping = queue.wake();
        UNUSED(sleeping);
        assert(!sleeping);
        schedule();
      }

      return prev;
    }

    /// Recursively raise the priority of the given cown and its blocker.
    static inline void
    backpressure_unblock(Cown* cown, Epoch epoch = Epoch(ThreadAlloc::get()))
    {
      UNUSED(epoch);
      for (; cown != nullptr;
           cown = cown->bp_state.load(std::memory_order_acquire).blocker())
      {
        Systematic::cout() << "Unblock cown " << cown << Systematic::endl;
        cown->backpressure_transition(Priority::High);
      }
    }

    /// Attempt to set the blocker for this cown. Return true if the priority is
    /// high. The blocker will not be set on a failed exchange due to another
    /// thread raising the priority of this cown.
    inline bool set_blocker(Cown* b)
    {
      auto bp = bp_state.load(std::memory_order_relaxed);
      yield();
      const auto success = bp_state.compare_exchange_strong(
        bp,
        BPState() | b | bp.priority() | bp.has_token(),
        std::memory_order_acq_rel);
      yield();
      assert(success || bp.high_priority());
      UNUSED(success);
      return bp.high_priority();
    }

    /// Return true if a sender to this cown should become low priority.
    inline bool triggers_muting()
    {
      auto p = bp_state.load(std::memory_order_acquire).priority();
      auto sleeping = queue.is_sleeping();
      yield();
      return (p != Priority::Normal) && !sleeping;
    }

    /// Set the `mutor` field of the current scheduler thread if the senders
    /// should be muted as a result of this message. Otherwise the `mutor`
    /// will remain null.
    static inline void
    backpressure_scan(const MessageBody& senders, const MessageBody& receivers)
    {
      if (Scheduler::local()->mutor != nullptr)
        return;

      // Ignore message if any senders are are in the set of receivers.
      for (size_t s = 0; s < senders.count; s++)
      {
        for (size_t r = 0; r < receivers.count; r++)
        {
          if (senders.cowns[s] == receivers.cowns[r])
            return;
        }
      }

      // Mute senders if any receivers are high or low priority.
      for (size_t r = 0; r < receivers.count; r++)
      {
        auto* receiver = receivers.cowns[r];
        if (receiver->triggers_muting() || coin(5))
        {
          assert(Scheduler::local()->mutor == nullptr);
          Scheduler::local()->mutor = receiver;
          Cown::acquire(receiver);
          return;
        }
      }
    }

    /// Return true if this cown's message queue is growing faster than the
    /// messages are being processed. This is based on an approximation for the
    /// length of this cown's message queue.
    inline bool overloaded() const
    {
      static constexpr size_t overload_rc = 800;
      const auto rc = get_header().rc.load(std::memory_order_relaxed);
      assert(rc != 0);
      yield();
      return rc >= overload_rc;
    }

    /// Update priority based on the occurrence of a token message, or replace
    /// the token if it is not in the queue. Return true if the current message
    /// is a token.
    inline bool check_token_message(Alloc* alloc, MessageBody* curr)
    {
      auto bp = bp_state.load(std::memory_order_acquire);

      auto set_has_token = [this, &bp](bool t) mutable {
        BPState next;
        do
        {
          assert(bp.has_token() != t);
          next = BPState() | bp.blocker() | bp.priority() | t;
          // When testing spurious failure, simulated by the `coin` returning
          // false, the exchange must not occur.
        } while (
          coin(9) ||
          !bp_state.compare_exchange_weak(bp, next, std::memory_order_acq_rel));
      };

      if (curr == nullptr)
      {
        Systematic::cout() << "Reached message token on cown " << this
                           << Systematic::endl;
        if (overloaded())
          return true;

        if (bp.priority() == Priority::High)
          backpressure_transition(Priority::MaybeHigh);
        else if (bp.priority() == Priority::MaybeHigh)
          backpressure_transition(Priority::Normal);

        set_has_token(false);
        return true;
      }

      if (!bp.has_token())
      {
        Systematic::cout() << "Cown " << this << ": enqueue message token"
                           << Systematic::endl;
        queue.enqueue(stub_msg(alloc));
        set_has_token(true);
      }

      return false;
    }

    inline bool check_unmute_message(Alloc* alloc, MessageBody* msg)
    {
      if (msg->behaviour != &unmute_behaviour)
        return false;

      for (size_t i = 0; i < msg->count; i++)
      {
        auto* cown = msg->cowns[i];
        if (cown == nullptr)
          break;

        Systematic::cout() << "Unmute cown " << cown << Systematic::endl;
        cown->backpressure_transition(Priority::Normal);
        Cown::release(alloc, cown);
      }

      alloc->dealloc(msg->cowns, msg->count * sizeof(Cown*));
      alloc->dealloc<sizeof(MessageBody)>(msg);

      return true;
    }

    /// Mute the senders participating in this message if a backpressure scan
    /// set the mutor during the behaviour. If false is returned, the caller
    /// must reschedule the senders and deallocate the senders array.
    inline bool apply_backpressure(
      Alloc* alloc, EpochMark epoch, Cown** senders, size_t count)
    {
      auto* mutor = Scheduler::local()->mutor;
      if (mutor == nullptr)
        return false;

      // The array of senders is reused for the unmute message. Since fewer than
      // the original count of cowns may be muted, a null terminator may be
      // added before the end of the allocation to mark the end of the muted
      // set.

      Scheduler::local()->mutor = nullptr;
      size_t muting_count = 0;
      for (size_t i = 0; i < count; i++)
      {
        auto* cown = senders[i];
        auto bp = cown->bp_state.load(std::memory_order_relaxed);
        const bool high_priority = bp.high_priority();
        yield();
        assert(bp.priority() != Priority::Low);

        // The cown may only be muted if its priority is normal and its epoch
        // mark is not `SCANNED`. Muting a scanned cown may result in the leak
        // detector collecting the cown while it is muted.
        if (high_priority || (cown->get_epoch_mark() == EpochMark::SCANNED))
        {
          cown->schedule();
          continue;
        }

        yield();
        Cown::acquire(cown);

        if (
#ifdef USE_SYSTEMATIC_TESTING
          Systematic::coin(9) ||
#endif
          !cown->bp_state.compare_exchange_weak(
            bp,
            BPState() | bp.blocker() | Priority::Low | bp.has_token(),
            std::memory_order_acq_rel))
        {
          assert(bp.priority() != Priority::Low);
          yield();
          cown->schedule();
          Cown::release(alloc, cown);
          continue;
        }

        Systematic::cout() << "Cown " << cown << ": backpressure state "
                           << bp.priority() << " -> Low" << Systematic::endl;
        assert(!high_priority);
        senders[muting_count++] = cown;
        Systematic::cout() << "Mute cown " << cown << " (mutor: cown " << mutor
                           << ")" << Systematic::endl;

        Scheduler::local()->mute_set_add(cown);
      }

      if (muting_count == 0)
      {
        alloc->dealloc(senders, count * sizeof(Cown*));
        Cown::release(alloc, mutor);
        return true;
      }

      if (muting_count < count)
        senders[muting_count] = nullptr;

      auto* msg = unmute_msg(alloc, count, senders, epoch);
      bool needs_scheduling = mutor->try_fast_send(msg);
      if (needs_scheduling)
        mutor->schedule();

      Cown::release(alloc, mutor);
      return true;
    }

    /**
     * This processes a batch of messages on a cown.
     *
     * It returns false, if the cown should not be rescheduled.
     *
     * It will process multi-messages and notifications.
     *
     * The notifications will only be processed once in a call to this.  It
     *will not process messages that were not in the queue before it began
     * processing messages.
     *
     * If this cown receives a notification after it has already called
     * cown_notified, then it guarantees to call cown_notified next time it is
     * called, and it is guaranteed to return true, so it will be rescheduled
     * or false if it is part of a multi-message acquire.
     **/
    bool run(Alloc* alloc, ThreadState::State, EpochMark epoch)
    {
      auto until = queue.peek_back();
      yield(); // Reading global state in peek_back().
      assert(
        bp_state.load(std::memory_order_acquire).priority() != Priority::Low);

      static constexpr size_t batch_limit = 100;
      auto notified_called = false;
      auto notify = false;

      MultiMessage* curr = nullptr;
      size_t batch_size = 0;
      do
      {
        assert(!queue.is_sleeping());

        curr = queue.dequeue(alloc, notify);

        if (!notified_called && notify)
        {
          notified_called = true;
          cown_notified();
        }

        if (curr == nullptr)
        {
          if (Scheduler::should_scan())
          {
            // We have hit null, and we should scan, then we know
            // all future messages must have been sent while in at least
            // pre-scan or have been counted.
            this->scan(ThreadAlloc::get(), Scheduler::local()->send_epoch);
            this->set_epoch_mark(Scheduler::local()->send_epoch);
          }

          // Reschedule if we have processed a message.
          // This is primarily an optimisation to keep busy cowns active cowns
          // around.
          // However,  if we remove this line then the leak detector will have
          // a bug.  It is possible to miss a wake-up from a Scan thread, is
          // the cown is currently active on a pre-scan thread. The following
          // should be added after if we alter this behaviour:
          //
          // // We are about to unschedule this cown, if another thread has
          // // marked this cown as scheduled for scan it will not have been
          // // able to reschedule it, but as this thread hasn't started
          // // scanning it will not have been scanned.  Ensure we can't miss
          // it
          // // by keeping in scheduler queue until the prescan phase has
          // // finished.
          // if (Scheduler::in_prescan())
          //   return true;
          //
          // TODO: Investigate systematic testing coverage here.
          if (batch_size != 0)
            return true;

          backpressure_transition(Priority::Normal, true);

          // Reschedule if cown does not go to sleep.
          if (!queue.mark_sleeping(notify))
          {
            if (notify)
            {
              // We must have run something to get here.
              assert(!notified_called);
              cown_notified();
              // Treat notification as a message and don't deschedule.
            }
            return true;
          }

          Systematic::cout() << "Unschedule cown " << this << Systematic::endl;
          Cown::release(alloc, this);
          return false;
        }

        assert(!queue.is_sleeping());
        auto* body = curr->get_body();

        if (check_token_message(alloc, body))
          return true;

        if (check_unmute_message(alloc, body))
          return true;

        batch_size++;

        Systematic::cout() << "Running Message " << curr << " on cown " << this
                           << Systematic::endl;

        auto* senders = body->cowns;
        const size_t senders_count = body->count;

        // A function that returns false indicates that the cown should not
        // be rescheduled, even if it has pending work. This also means the
        // cown's queue should not be marked as empty, even if it is.
        if (!run_step(curr))
          return false;

        if (apply_backpressure(alloc, epoch, senders, senders_count))
          return false;

        // Reschedule the other cowns.
        for (size_t s = 0; s < (senders_count - 1); s++)
          senders[s]->schedule();

        alloc->dealloc(senders, senders_count * sizeof(Cown*));

      } while ((curr != until) && (batch_size < batch_limit));

      return true;
    }

    bool try_collect(Alloc* alloc, EpochMark epoch)
    {
      Systematic::cout() << "try_collect: " << this << " (" << get_epoch_mark()
                         << ")" << Systematic::endl;

      if (in_epoch(EpochMark::SCHEDULED_FOR_SCAN))
      {
        Systematic::cout() << "Clearing SCHEDULED_FOR_SCAN state: " << this
                           << Systematic::endl;
        // There is a race, when multiple threads may attempt to
        // schedule a Cown for tracing.  In this case, we can
        // get a stale descriptor mark. Update it here, for the
        // next LD.
        set_epoch_mark(epoch);
        return false;
      }

      if (in_epoch(epoch))
        return false;

      // Check if the Cown is already collected.
      if (!is_collected())
      {
        yield();
        assert(
          bp_state.load(std::memory_order_acquire).priority() != Priority::Low);
        Systematic::cout() << "Collecting (sweep) cown " << this
                           << Systematic::endl;
        collect(alloc);
      }

      return true;
    }

    inline bool is_live(EpochMark send_epoch)
    {
      return in_epoch(EpochMark::SCHEDULED_FOR_SCAN) || in_epoch(send_epoch);
    }

    /**
     * Called when strong reference count reaches one.
     * Uses thread_local state to deal with deep deallocation
     * chains by queuing recursive calls.
     **/
    void queue_collect(Alloc* alloc)
    {
      thread_local ObjectStack* work_list = nullptr;

      // If there is a already a queue, use it
      if (work_list != nullptr)
      {
        work_list->push(this);
        return;
      }

      // Make queue for recursive deallocations.
      ObjectStack current(alloc);
      work_list = &current;

      // Collect the current cown
      collect(alloc);
      yield();
      weak_release(alloc);

      // Collect recursively reachable cowns
      while (!current.empty())
      {
        auto a = (Cown*)current.pop();
        a->collect(alloc);
        yield();
        a->weak_release(alloc);
      }
      work_list = nullptr;
    }

    void collect(Alloc* alloc)
    {
      // If this was collected by leak detector, then don't double dealloc
      // cown body, when the ref count drops.
      if (is_collected())
        return;

      mark_collected();

#ifdef USE_SYSTEMATIC_TESTING_WEAK_NOTICEBOARDS
      flush_all(alloc);
#endif
      Systematic::cout() << "Collecting cown " << this << Systematic::endl;

      ObjectStack dummy(alloc);
      // Run finaliser before releasing our data.
      // Sub-regions handled by code below.
      finalise(nullptr, dummy);

      // Release our data.
      ObjectStack f(alloc);
      trace(f);

      while (!f.empty())
      {
        Object* o = f.pop();

        switch (o->get_class())
        {
          case RegionMD::ISO:
            Region::release(alloc, o);
            break;

          case RegionMD::RC:
          case RegionMD::SCC_PTR:
            Immutable::release(alloc, o);
            break;

          case RegionMD::COWN:
            Systematic::cout()
              << "DecRef from " << this << " to " << o << Systematic::endl;
            Cown::release(alloc, (Cown*)o);
            break;

          default:
            abort();
        }
      }

      yield();
      assert(
        bp_state.load(std::memory_order_acquire).priority() != Priority::Low);

      // Now we may run our destructor.
      destructor();

      auto* stub = queue.destroy();
      // All messages must have been run by the time the cown is collected.
      assert(stub->next.load(std::memory_order_relaxed) == nullptr);

      alloc->dealloc<sizeof(MultiMessage)>(stub);
    }

    /**
     * Create a `MultiMessage` that is never sent or processed.
     */
    static MultiMessage* stub_msg(Alloc* alloc)
    {
      return MultiMessage::make_message(alloc, nullptr, EpochMark::EPOCH_NONE);
    }

    /**
     * Create an unmute message using an empty behaviour. The given array of
     * cowns may be null terminated, but the count must always be count of
     * pointers that indicates the size of the allocation.
     */
    static MultiMessage*
    unmute_msg(Alloc* alloc, size_t count, Cown** cowns, EpochMark epoch)
    {
      auto* body =
        MultiMessage::make_body(alloc, count, cowns, &unmute_behaviour);
      return MultiMessage::make_message(alloc, body, epoch);
    }
  };

  namespace cown
  {
    inline void release(Alloc* alloc, Cown* o)
    {
      Cown::release(alloc, o);
    }

    inline void mark_for_scan(Object* o, EpochMark epoch)
    {
      Cown::mark_for_scan(o, epoch);
    }
  } // namespace cown
} // namespace verona::rt

namespace Systematic
{
  inline std::string get_systematic_id()
  {
#if defined(USE_SYSTEMATIC_TESTING) || defined(USE_FLIGHT_RECORDER)
    static std::atomic<size_t> external_id_source = 1;
    static thread_local size_t external_id = 0;
    auto s = verona::rt::Scheduler::local();
    if (s != nullptr)
    {
      std::stringstream ss;
      auto offset = static_cast<int>(s->systematic_id % 9);
      if (offset != 0)
        ss << std::setw(offset) << " ";
      ss << s->systematic_id;
      ss << std::setw(9 - offset) << " ";
      return ss.str();
    }
    if (external_id == 0)
    {
      auto e = external_id_source.fetch_add(1);
      external_id = e;
    }
    std::stringstream ss;
    bool short_id = external_id <= 26;
    size_t spaces = short_id ? 9 : 8;
    size_t offset = (external_id - 1) % spaces;
    if (offset != 0)
      ss << std::setw(spaces - offset) << " ";
    if (short_id)
      ss << (char)('a' + (external_id - 1));
    else
      ss << 'E' << (external_id - 26);
    ss << std::setw(offset) << " ";
    return ss.str();
#else
    return "";
#endif
  }
}
