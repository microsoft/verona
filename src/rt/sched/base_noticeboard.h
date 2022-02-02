// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../ds/forward_list.h"
#include "../region/region.h"
#include "../sched/epoch.h"
#include "../sched/schedulerthread.h"
#include "../test/logging.h"

#include <queue>

namespace verona::rt
{
  class Cown;
  using CownThread = SchedulerThread<Cown>;
  using Scheduler = ThreadPool<CownThread>;

  class BaseNoticeboard
  {
    using CT = std::conditional_t<
      (sizeof(uintptr_t) > sizeof(uint64_t)),
      uintptr_t,
      uint64_t>;

  private:
    // The content of a noticeboard; only accessible via `put` and `get`.
    unsigned char content[sizeof(CT)];

  protected:
    // Indicate if the content of the noticeboard is std::is_fundamental
    bool is_fundamental;

    template<typename T>
    void put(T v)
    {
      static_assert(sizeof(T) <= sizeof(CT));
      *(T*)content = v;
    }

    template<typename T>
    T get() const
    {
      static_assert(sizeof(T) <= sizeof(CT));
      return *(T*)content;
    }

#ifdef USE_SYSTEMATIC_TESTING_WEAK_NOTICEBOARDS
    std::deque<CT> update_buffer;

    template<typename T>
    void update_buffer_push(T v)
    {
      update_buffer.push_back((CT)v);
    }

    void flush_n(Alloc& alloc, size_t n)
    {
      assert(n > 0);

      if (is_fundamental)
      {
        (void)alloc;
        auto prev = get<uint64_t>();
        for (size_t i = 0; i < n; ++i)
        {
          assert(!update_buffer.empty());
          prev = update_buffer.front();
          update_buffer.pop_front();
        }
        put(prev);
      }
      else
      {
        Epoch e(alloc);
        auto prev = get<Object*>();
        for (size_t i = 0; i < n; ++i)
        {
          e.dec_in_epoch(prev);
          assert(!update_buffer.empty());
          prev = (Object*)update_buffer.front();
          update_buffer.pop_front();
        }
        assert(prev);
        put(prev);
      }
    }

  public:
    void flush_all(Alloc& alloc)
    {
      if (update_buffer.empty())
      {
        return;
      }
      Logging::cout() << "Flushing values on noticeboard: " << this
                      << Logging::endl;
      flush_n(alloc, update_buffer.size());
    }

    void flush_some(Alloc& alloc)
    {
      if (update_buffer.empty())
      {
        return;
      }
      auto n = update_buffer.size();
      auto r = Systematic::get_prng_next();
      auto pick = (size_t)(r % (n + 1));
      if (pick == 0)
      {
        return;
      }
      flush_n(alloc, pick);
    }
#endif
  };
} // namespace verona::rt
