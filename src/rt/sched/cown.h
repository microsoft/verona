// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../ds/forward_list.h"
#include "../ds/mpscq.h"
#include "../region/region.h"
#include "../test/logging.h"
#include "../test/systematic.h"
#include "base_noticeboard.h"
#include "multimessage.h"
#include "schedulerthread.h"

#include <algorithm>

namespace verona::rt
{
  using namespace snmalloc;
  class Cown;
  using CownThread = SchedulerThread<Cown>;
  using Scheduler = ThreadPool<CownThread>;

  static void yield()
  {
    Systematic::yield();
  }

  static Behaviour unmute_behaviour{Behaviour::Descriptor::empty()};

  struct EnqueueLock
  {
    std::atomic<bool> locked = false;

    void lock()
    {
      auto u = false;
      while (!locked.compare_exchange_strong(u, true))
      {
        u = false;
        while (locked)
        {
          yield();
        }
      }
    }

    void unlock()
    {
      locked.store(false, std::memory_order_release);
    }
  };

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
        auto& alloc = ThreadAlloc::get();
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
    friend class MultiMessage;
    friend CownThread;

    template<typename T>
    friend class Noticeboard;

    template<typename T>
    friend class MPMCQ;

    static constexpr auto NO_EPOCH_SET = (std::numeric_limits<uint64_t>::max)();

    union
    {
      std::atomic<Cown*> next_in_queue;
      uint64_t epoch_when_popped{NO_EPOCH_SET};
    };

    // Seven pointer overhead compared to an object.
    verona::rt::MPSCQ<MultiMessage> queue{};

    // Used for garbage collection of cyclic cowns only.
    // Uses the bottom bit to indicate the cown has been collected
    // If the object is collected by the leak detector, we should not
    // collect again when the weak reference count hits 0.
    std::atomic<uintptr_t> thread_status{0};
    Cown* next{nullptr};

    /**
     * Cown's weak reference count.  This keeps the cown itself alive, but not
     * the data it can reach.  Weak reference can be promoted to strong, if a
     * strong reference still exists.
     **/
    std::atomic<size_t> weak_count{1};

    EnqueueLock enqueue_lock;

    static Cown* create_token_cown()
    {
      static constexpr Descriptor desc = {
        vsizeof<Cown>, nullptr, nullptr, nullptr};
      auto p = ThreadAlloc::get().alloc<desc.size>();
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

    void flush_all(Alloc& alloc)
    {
      for (auto b : noticeboards)
      {
        b->flush_all(alloc);
      }
    }

    void flush_some(Alloc& alloc)
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
      Logging::cout() << "Cown " << o << " acquire" << Logging::endl;
      assert(o->debug_is_cown());
      o->incref();
    }

    static void release(Alloc& alloc, Cown* o)
    {
      Logging::cout() << "Cown " << o << " release" << Logging::endl;
      assert(o->debug_is_cown());
      Cown* a = ((Cown*)o);

      // Perform decref
      bool last = o->decref_cown();
      yield();

      if (!last)
        return;

      // All paths from this point must release the weak count owned by the
      // strong count.

      Logging::cout() << "Cown " << o << " dealloc" << Logging::endl;

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
          Logging::cout() << "Not performing recursive deallocation on: " << o
                          << Logging::endl;
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
    void weak_release(Alloc& alloc)
    {
      Logging::cout() << "Cown " << this << " weak release" << Logging::endl;
      if (weak_count.fetch_sub(1) == 1)
      {
        auto* t = owning_thread();
        yield();
        if (!t)
        {
          // Deallocate an unowned cown
          Logging::cout()
            << "Not allocated on a Verona thread, so deallocating: " << this
            << Logging::endl;
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
      Logging::cout() << "Cown " << this << " weak acquire" << Logging::endl;
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
        Logging::cout() << "Already marked " << cown << " ("
                        << cown->get_epoch_mark() << ")" << Logging::endl;
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

    void dealloc(Alloc& alloc)
    {
      Object::dealloc(alloc);
      yield();
    }

    bool scanned(EpochMark epoch)
    {
      return in_epoch(epoch);
    }

    void scan(Alloc& alloc, EpochMark epoch)
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

    static void scan_stack(Alloc& alloc, EpochMark epoch, ObjectStack& f)
    {
      while (!f.empty())
      {
        Object* o = f.pop();
        switch (o->get_class())
        {
          case RegionMD::ISO:
            Logging::cout()
              << "Object Scan: reaches region: " << o << Logging::endl;
            Region::cown_scan(alloc, o, epoch);
            break;

          case RegionMD::RC:
          case RegionMD::SCC_PTR:
            Logging::cout()
              << "Object Scan: reaches immutable: " << o << Logging::endl;
            Immutable::mark_and_scan(alloc, o, epoch);
            break;

          case RegionMD::COWN:
            Logging::cout()
              << "Object Scan: reaches cown " << o << Logging::endl;
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
      auto& alloc = ThreadAlloc::get();
      const auto last = body->count - 1;
      assert(body->index <= last);

      // First acquire all the locks if a multimessage
      if (body->count > 1)
      {
        for (size_t i = 0; i < body->count; i++)
        {
          auto* next = body->cowns[i];
          Logging::cout() << "Will try to acquire lock " << next
                          << Logging::endl;
          next->enqueue_lock.lock();
          yield();
          Logging::cout() << "Acquired lock " << next << Logging::endl;
        }
      }

      size_t loop_end = body->count;
      for (size_t i = 0; i < loop_end; i++)
      {
        auto m = MultiMessage::make_message(alloc, body, epoch);
        auto* next = body->cowns[i];
        Logging::cout() << "MultiMessage " << m << ": fast requesting " << next
                        << ", index " << i << " behaviour " << body->behaviour
                        << " loop end " << loop_end << Logging::endl;

        auto needs_sched = next->try_fast_send(m);
        if (loop_end > 1)
          next->enqueue_lock.unlock();

        if (!needs_sched)
        {
          Logging::cout() << "try fast send found busy cown " << body
                          << " loop iteration " << i << " cown " << next
                          << Logging::endl;
          continue;
        }

        Logging::cout() << "Will schedule cown " << next << Logging::endl;
        if (i == last)
        {
          next->schedule();
          return;
        }

        body->exec_count_down.fetch_sub(1);

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
      Logging::cout() << "Enqueue MultiMessage " << m << Logging::endl;
      bool needs_scheduling = queue.enqueue(m);
      Logging::cout() << "Enqueued MultiMessage " << m << " needs scheduling? "
                      << needs_scheduling << Logging::endl;
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
    bool run_step(MultiMessage* m)
    {
      MultiMessage::MultiMessageBody& body = *(m->get_body());
      Alloc& alloc = ThreadAlloc::get();

      EpochMark e = m->get_epoch();

      Logging::cout() << "MultiMessage " << m << " index " << body.index
                      << " acquired " << this << " epoch " << e
                      << Logging::endl;

      // If we are in should_scan, and we observe a message in this epoch,
      // then all future messages must have been sent while in pre-scan or
      // later. Thus any messages that weren't implicitly scanned on send,
      // will be counted as inflight
      if (Scheduler::should_scan() && e == Scheduler::local()->send_epoch)
      {
        // TODO: Investigate systematic testing coverage here.
        if (get_epoch_mark() != Scheduler::local()->send_epoch)
        {
          scan(alloc, Scheduler::local()->send_epoch);
          set_epoch_mark(Scheduler::local()->send_epoch);
        }
      }

      if (body.exec_count_down.fetch_sub(1) > 1)
      {
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
          Logging::cout() << "Trace message: " << m << Logging::endl;

          // Scan cowns for this message, as they may not have been scanned
          // yet.
          for (size_t i = 0; i < body.count; i++)
          {
            Logging::cout()
              << "Scanning cown " << body.cowns[i] << Logging::endl;
            body.cowns[i]->scan(alloc, Scheduler::local()->send_epoch);
          }

          // Scan closure
          ObjectStack f(alloc);
          body.behaviour->trace(f);
          scan_stack(alloc, Scheduler::local()->send_epoch, f);
        }
        else
        {
          Logging::cout() << "Trace message not required: " << m << " (" << e
                          << ")" << Logging::endl;
        }
      }

      Scheduler::local()->message_body = &body;

      // Run the behaviour.
      body.behaviour->f();

      for (size_t i = 0; i < body.count; i++)
      {
        if (body.cowns[i])
          Cown::release(alloc, body.cowns[i]);
      }

      Logging::cout() << "MultiMessage " << m << " completed and running on "
                      << this << Logging::endl;

      // Free the body and the behaviour.
      alloc.dealloc(body.behaviour, body.behaviour->size());
      alloc.dealloc<sizeof(MultiMessage::MultiMessageBody)>(m->get_body());

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
      Logging::cout() << "Schedule behaviour of type: " << typeid(Be).name()
                      << Logging::endl;

      auto& alloc = ThreadAlloc::get();
      auto* be =
        new ((Be*)alloc.alloc<sizeof(Be)>()) Be(std::forward<Args>(args)...);
      auto** sort = (Cown**)alloc.alloc(count * sizeof(Cown*));
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

      // Try to acquire as many cowns as possible without rescheduling,
      // starting from the beginning.
      fast_send(body, epoch);
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
    bool run(Alloc& alloc, ThreadState::State)
    {
      auto until = queue.peek_back();
      yield(); // Reading global state in peek_back().

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

          // We are about to unschedule this cown, if another thread has
          // marked this cown as scheduled for scan it will not have been
          // able to reschedule it, but as this thread hasn't started
          // scanning it will not have been scanned.  Ensure we can't miss it
          // by keeping in scheduler queue until the prescan phase has
          // finished.
          if (Scheduler::in_prescan())
            return true;

          // Reschedule if we have processed a message.
          // This is primarily an optimisation to keep busy cowns active cowns
          // around.
          // TODO The following could be removed to improve the single action
          // cown case.
          //      This is designed to be effective if a cown is receiving a lot
          //      of messages.
          if (batch_size != 0)
            return true;

          // Reschedule if cown does not go to sleep.
          if (!queue.mark_sleeping(alloc, notify))
          {
            if (notify)
            {
              // It is possible to have already notified the cown in this batch,
              // but the notification on the mark_sleeping could have occurred
              // after the previous call to cown_notified, so need to call
              // again.
              cown_notified();
              // Don't deschedule send round again.  We could try to
              // mark_sleeping but that could lead to another notification
              // having been received, and then we wouldn't process anything
              // else.
            }
            return true;
          }

          Logging::cout() << "Unschedule cown " << this << Logging::endl;
          Cown::release(alloc, this);
          return false;
        }

        assert(!queue.is_sleeping());
        auto* body = curr->get_body();

        batch_size++;

        Logging::cout() << "Running Message " << curr << " on cown " << this
                        << Logging::endl;

        auto* senders = body->cowns;
        const size_t senders_count = body->count;

        // A function that returns false indicates that the cown should not
        // be rescheduled, even if it has pending work. This also means the
        // cown's queue should not be marked as empty, even if it is.
        if (!run_step(curr))
          return false;

        // Reschedule the other cowns.
        for (size_t s = 0; s < senders_count; s++)
        {
          if ((senders[s]) && (senders[s] != this))
            senders[s]->schedule();
        }

        alloc.dealloc(senders, senders_count * sizeof(Cown*));

      } while ((curr != until) && (batch_size < batch_limit));

      return true;
    }

    bool try_collect(Alloc& alloc, EpochMark epoch)
    {
      Logging::cout() << "try_collect: " << this << " (" << get_epoch_mark()
                      << ")" << Logging::endl;

      if (in_epoch(EpochMark::SCHEDULED_FOR_SCAN))
      {
        Logging::cout() << "Clearing SCHEDULED_FOR_SCAN state: " << this
                        << Logging::endl;
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
        Logging::cout() << "Collecting (sweep) cown " << this << Logging::endl;
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
    void queue_collect(Alloc& alloc)
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

    void collect(Alloc& alloc)
    {
      // If this was collected by leak detector, then don't double dealloc
      // cown body, when the ref count drops.
      if (is_collected())
        return;

      mark_collected();

#ifdef USE_SYSTEMATIC_TESTING_WEAK_NOTICEBOARDS
      flush_all(alloc);
#endif
      Logging::cout() << "Collecting cown " << this << Logging::endl;

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
            Logging::cout()
              << "DecRef from " << this << " to " << o << Logging::endl;
            Cown::release(alloc, (Cown*)o);
            break;

          default:
            abort();
        }
      }

      yield();

      // Now we may run our destructor.
      destructor();

      auto* stub = queue.destroy();
      // All messages must have been run by the time the cown is collected.
      assert(stub->next.load(std::memory_order_relaxed) == nullptr);

      alloc.dealloc<sizeof(MultiMessage)>(stub);
    }

    bool release_early()
    {
      auto* body = Scheduler::local()->message_body;
      auto* senders = body->cowns;
      const size_t senders_count = body->count;
      Alloc& alloc = ThreadAlloc::get();

      /*
       * Avoid releasing the last cown because it breaks the current
       * code structure
       */
      if (this == senders[senders_count - 1])
        return false;

      for (size_t s = 0; s < senders_count; s++)
      {
        if (senders[s] != this)
          continue;
        Cown::release(alloc, senders[s]);
        senders[s]->schedule();
        senders[s] = nullptr;
        break;
      }

      return true;
    }

    /**
     * Create a `MultiMessage` that is never sent or processed.
     */
    static MultiMessage* stub_msg(Alloc& alloc)
    {
      return MultiMessage::make_message(alloc, nullptr, EpochMark::EPOCH_NONE);
    }

    /**
     * Create an unmute message using an empty behaviour. The given array of
     * cowns may be null terminated, but the count must always be count of
     * pointers that indicates the size of the allocation.
     */
    static MultiMessage*
    unmute_msg(Alloc& alloc, size_t count, Cown** cowns, EpochMark epoch)
    {
      auto* body =
        MultiMessage::make_body(alloc, count, cowns, &unmute_behaviour);
      return MultiMessage::make_message(alloc, body, epoch);
    }
  };

  namespace cown
  {
    inline void release(Alloc& alloc, Cown* o)
    {
      Cown::release(alloc, o);
    }

    inline void mark_for_scan(Object* o, EpochMark epoch)
    {
      Cown::mark_for_scan(o, epoch);
    }
  } // namespace cown
} // namespace verona::rt

namespace Logging
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
    unsigned char spaces = short_id ? 9 : 8;
    // Modulo guarantees that this fits into the same type as spaces.
    decltype(spaces) offset =
      static_cast<decltype(spaces)>((external_id - 1) % spaces);
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
