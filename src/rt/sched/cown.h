// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../ds/forward_list.h"
#include "../ds/mpscq.h"
#include "../region/region.h"
#include "../test/logging.h"
#include "../test/systematic.h"
#include "base_noticeboard.h"
#include "core.h"
#include "multimessage.h"
#include "schedulerthread.h"

#include <algorithm>
#include <vector>

namespace verona::rt
{
  using namespace snmalloc;
  class Cown;
  using CownThread = SchedulerThread<Cown>;
  using Scheduler = ThreadPool<CownThread, Cown>;

  static void yield()
  {
    Systematic::yield();
  }

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
   * accessed by a single (scheduler) thread at a time when writing, or
   * accessed by multiple (scheduler) threads at a time when reading.
   * A cown can only be in one of the following states:
   *   1. Unscheduled
   *   2. Scheduled, in the queue of a single scheduler thread
   *   3. Running with read/write access on a single scheduler thread, and not
   * in the queue of any scheduler thread
   *   4. Running with read access on one or more scheduler threads, and may
   * also be in the queue of one other scheduler thread
   *
   * Once a cown is running, it executes a batch of multi-message behaviours.
   * Each message may either acquire the running cown for participation in a
   * future behaviour, or execute the behaviour if it is the last cown to be
   * acquired.
   * If the running cown is acquired for writing for a future behaviour, it will
   * be descheduled until that behaviour has completed. If the running cown is
   * acquired for reading for a future behaviour, it will not be descheduled. If
   * the running cown is acquired for reading _and_ executing on this thread,
   * the cown will be rescheduled to be picked up by another thread. (it might
   * later return to this thread if this is the last thread to use the cown in
   * read more before a write).
   */

  struct ReadRefCount
  {
    std::atomic<size_t> count{0};

    void add_read()
    {
      count.fetch_add(2);
    }

    // true means last reader and writer is waiting, false otherwise
    bool release_read()
    {
      if (count.fetch_sub(2) == 3)
      {
        Systematic::yield();
        assert(count.load() == 1);
        count.store(0, std::memory_order_relaxed);
        return true;
      }
      return false;
    }

    bool try_write()
    {
      if (count.load(std::memory_order_relaxed) == 0)
        return true;

      // Mark a pending write
      if (count.fetch_add(1) != 0)
        return false;

      // if in the time between reading and writing the ref count, it
      // became zero, we can now process the write, so clear the flag
      // and continue
      count.fetch_sub(1);
      Systematic::yield();
      assert(count.load() == 0);
      return true;
    }
  };

  class Cown : public Object
  {
    using MessageBody = MultiMessage::Body;

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
        queue.init(stub_msg(alloc));
      }
    }

  private:
    friend class MultiMessage;
    friend CownThread;
    friend Core<Cown>;

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

    /**
     * Cown's weak reference count.  This keeps the cown itself alive, but not
     * the data it can reach.  Weak reference can be promoted to strong, if a
     * strong reference still exists.
     **/
    std::atomic<size_t> weak_count{1};

    /*
     * Cown's read ref count.
     * Bottom bit is used to signal a waiting write.
     * Remaining bits are the count.
     */
    ReadRefCount read_ref_count;

    EnqueueLock enqueue_lock;

    static const Descriptor* token_desc()
    {
      static constexpr Descriptor desc{
        vsizeof<Cown>, nullptr, nullptr, nullptr};
      return &desc;
    }

    static Cown* create_token_cown()
    {
      auto desc = token_desc();
      auto p = ThreadAlloc::get().alloc(desc->size);
      auto o = Object::register_object(p, desc);
      auto a = new (o) Cown(false);
      return a;
    }

    void set_token_owning_core(Core<Cown>* owner)
    {
      assert(get_descriptor() == token_desc());
      // Use region meta_data to point at the
      // owning scheduler thread for a token cown.
      init_next((Object*)owner);
    }

    Core<Cown>* get_token_owning_core()
    {
      assert(get_descriptor() == token_desc());
      return (Core<Cown>*)(get_next());
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
      auto release_weak = false;
      bool last = o->decref_cown(release_weak);

      yield();

      if (release_weak)
      {
        o->weak_release(alloc);
        yield();
      }

      if (!last)
        return;

      // All paths from this point must release the weak count owned by the
      // strong count.

      Logging::cout() << "Cown " << o << " dealloc" << Logging::endl;

      // If last, then collect the cown body.
      a->queue_collect(alloc);
    }

    /**
     * Release a weak reference to this cown.
     **/
    void weak_release(Alloc& alloc)
    {
      Logging::cout() << "Cown " << this << " weak release" << Logging::endl;
      if (weak_count.fetch_sub(1) == 1)
      {
        yield();

        Logging::cout() << "Cown " << this << " no references left."
                        << Logging::endl;
        auto epoch = epoch_when_popped;
        auto outdated =
          epoch == NO_EPOCH_SET || GlobalEpoch::is_outdated(epoch);
        if (outdated || Scheduler::is_teardown_in_progress())
        {
          // If teardown is in progress, then there are no ABA issues, so
          // directly deallocate.
          // TODO: Investigate if we could eject global epoch to avoid the
          // additional test.
          Logging::cout() << "Cown " << this << " dealloc from weak_release"
                          << Logging::endl;
          dealloc(alloc);
        }
        else
        {
          Logging::cout() << "Cown " << this << " defer dealloc"
                          << Logging::endl;
          // There could be an ABA problem if we reuse this cown as the epoch
          // has not progressed enough We delay the deallocation until the epoch
          // has progressed enough
          // TODO: We are waiting too long as this is inserting in the current
          // epoch, and not `epoch` which is all that is required.
          Epoch e(alloc);
          e.delete_in_epoch(this->real_start());
        }
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
      bool reacquire_weak = false;
      auto result = Object::acquire_strong_from_weak(reacquire_weak);
      if (reacquire_weak)
      {
        weak_acquire();
      }
      return result;
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

      auto* core = Scheduler::round_robin();
      CownThread::schedule_lifo(core, this);
    }

  private:
    void dealloc(Alloc& alloc)
    {
      Object::dealloc(alloc);
      yield();
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
    template<TransferOwnership transfer = NoTransfer>
    static void fast_send(MultiMessage::Body* body)
    {
      auto& alloc = ThreadAlloc::get();
      const auto last = body->count - 1;

      // First acquire all the locks if a multimessage
      if (body->count > 1)
      {
        for (size_t i = 0; i < body->count; i++)
        {
          auto next = body->get_requests_array()[i];
          Logging::cout() << "Will try to acquire lock " << next.cown()
                          << Logging::endl;
          next.cown()->enqueue_lock.lock();
          yield();
          Logging::cout() << "Acquired lock " << next.cown() << Logging::endl;
        }
      }

      size_t loop_end = body->count;
      for (size_t i = 0; i < loop_end; i++)
      {
        auto m = MultiMessage::make_message(alloc, body);
        auto next = body->get_requests_array()[i].cown();
        Logging::cout() << "MultiMessage " << m << ": fast requesting " << next
                        << ", index " << i << " loop end " << loop_end
                        << Logging::endl;

        auto needs_sched = next->try_fast_send<transfer>(m);
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

        Request r = body->get_requests_array()[i];
        if (r.is_read())
        {
          r.cown()->read_ref_count.add_read();
          r.cown()->schedule();
        }
      }
    }

    /**
     * This method implements an optimized multi-message send to a cown. A
     * sleeping cown will not be reschdeuled because we want to immediately
     * acquire the cown without going through the scheduler queue. Returns true
     * if the cown was asleep and needs scheduling; returns false otherwise.
     **/
    template<TransferOwnership transfer = NoTransfer>
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
      if (needs_scheduling && transfer == NoTransfer)
      {
        Cown::acquire(this);
      }
      if (!needs_scheduling && transfer == YesTransfer)
      {
        Cown::release(ThreadAlloc::get(), this);
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
      MultiMessage::Body& body = *(m->get_body());
      Alloc& alloc = ThreadAlloc::get();

      Logging::cout() << "MultiMessage " << m << " acquired " << this
                      << Logging::endl;

      // Find this cown in the list of cowns to acquire
      Request* request = body.get_requests_array();
      while (request->cown() != this)
        request++;

      bool schedule_after_behaviour = true;
      if (request->is_read())
      {
        read_ref_count.add_read();
        if (body.exec_count_down.fetch_sub(1) > 1)
          return true;
        else
          // In this case, this thread will execute the behaviour
          // but another thread can still read this cown, so reschedule
          // it before executing the behaviour and do not reschedule it after.
          schedule_after_behaviour = false;
      }
      else
      { // request->mode == AccessMode::WRITE
        if (body.exec_count_down.fetch_sub(1) > 1)
          return false;
      }

      Scheduler::local()->message_body = &body;

      // The message `m` will be deallocated when the next scheduler thread
      // picks up a message, so wait until after all the possible uses of `m` to
      // reschedule this cown.
      if (!schedule_after_behaviour)
        schedule();

      // Run the behaviour.
      body.get_behaviour().f();

      Logging::cout() << "MultiMessage " << m << " completed and running on "
                      << this << Logging::endl;

      //  Reschedule the writeable cowns (read-only cowns are not unscheduled
      //  for a behaviour).
      for (size_t s = 0; s < body.count; s++)
      {
        Request request = body.get_requests_array()[s];
        Cown* cown = request.cown();
        if (cown)
        {
          if (!request.is_read() && cown != this)
            cown->schedule();
          else if (request.is_read())
          {
            // If this read is the last read of a cown, and that cown's next
            // message requires writing, clear the flag and either:
            //   - schedule the cown if it is not this cown, or
            //   - continue processing this cown's message on this scheduler
            //   thread
            //     (overwriting the previous decision to stop processing this
            //     cown).
            if (cown->read_ref_count.release_read())
            {
              if (cown != this)
                cown->schedule();
              else
                schedule_after_behaviour = true;
            }
          }
        }
      }

      alloc.dealloc(&body);

      return schedule_after_behaviour;
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

    template<
      class Behaviour,
      TransferOwnership transfer = NoTransfer,
      typename... Args>
    static void schedule(size_t count, Cown** cowns, Args&&... args)
    {
      // TODO Remove vector allocation here.  This is a temporary fix to
      // as we transition to using Request through the code base.
      auto& alloc = ThreadAlloc::get();
      Request* requests = (Request*)alloc.alloc(count * sizeof(Request));

      for (size_t i = 0; i < count; ++i)
        requests[i] = Request::write(cowns[i]);

      schedule<Behaviour, transfer, Args...>(
        count, requests, std::forward<Args>(args)...);

      alloc.dealloc(requests);
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
    static void schedule(size_t count, Request* requests, Args&&... args)
    {
      static_assert(std::is_base_of_v<Behaviour, Be>);
      Logging::cout() << "Schedule behaviour of type: " << typeid(Be).name()
                      << Logging::endl;

      auto& alloc = ThreadAlloc::get();

      auto body =
        MultiMessage::Body::make<Be>(alloc, count, std::forward<Args>(args)...);

      auto* sort = body->get_requests_array();
      memcpy(sort, requests, count * sizeof(Request));

#ifdef USE_SYSTEMATIC_TESTING
      std::sort(&sort[0], &sort[count], [](Request& a, Request& b) {
        return a.cown()->id() < b.cown()->id();
      });
#else
      std::sort(&sort[0], &sort[count], [](Request& a, Request& b) {
        return a.cown() < b.cown();
      });
#endif

      // Try to acquire as many cowns as possible without rescheduling,
      // starting from the beginning.
      fast_send<transfer>(body);
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

    bool run(Alloc& alloc)
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

        curr = queue.peek();

        if (curr != nullptr)
        {
          auto* body = curr->get_body();

          // find us in the list of cowns to acquire
          Request* request = body->get_requests_array();
          while (request->cown() != this)
            request++;

          // Attempt to process a write, if it fails stop processing the message
          // queue.
          if (!request->is_read() && !read_ref_count.try_write())
            return false;
        }

        curr = queue.dequeue(alloc, notify);

        if (!notified_called && notify)
        {
          notified_called = true;
          cown_notified();
        }

        if (curr == nullptr)
        {
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

        batch_size++;

        Logging::cout() << "Running Message " << curr << " on cown " << this
                        << Logging::endl;

        // A function that returns false indicates that the cown should not
        // be rescheduled, even if it has pending work. This also means the
        // cown's queue should not be marked as empty, even if it is.
        if (!run_step(curr))
          return false;

      } while ((curr != until) && (batch_size < batch_limit));

      return true;
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
        // This is a recursive call, add to queue
        // and return immediately.
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
#ifdef USE_SYSTEMATIC_TESTING_WEAK_NOTICEBOARDS
      flush_all(alloc);
#endif
      Logging::cout() << "Collecting cown " << this << Logging::endl;

      ObjectStack dummy(alloc);
      // Run finaliser before releasing our data.
      // Sub-regions handled by code below.
      finalise(nullptr, dummy);

      // TODO?
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
      auto* senders = body->get_requests_array();
      const size_t senders_count = body->count;
      Alloc& alloc = ThreadAlloc::get();

      /*
       * Avoid releasing the last cown because it breaks the current
       * code structure
       */
      if (this == senders[senders_count - 1].cown())
        return false;

      for (size_t s = 0; s < senders_count; s++)
      {
        if (senders[s].cown() != this)
          continue;

        if (
          !senders[s].is_read() ||
          senders[s].cown()->read_ref_count.release_read())
        {
          senders[s].cown()->schedule();
        }
        else
        {
          Cown::release(alloc, senders[s].cown());
        }

        senders[s] = Request();

        break;
      }

      return true;
    }

    /**
     * Create a `MultiMessage` that is never sent or processed.
     */
    static MultiMessage* stub_msg(Alloc& alloc)
    {
      return MultiMessage::make_message(alloc, nullptr);
    }
  };

  namespace cown
  {
    inline void release(Alloc& alloc, Cown* o)
    {
      Cown::release(alloc, o);
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
