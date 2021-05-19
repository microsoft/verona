// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../ds/forward_list.h"
#include "../region/region.h"
#include "../sched/epoch.h"
#include "../sched/schedulerthread.h"
#include "../test/systematic.h"

#include <queue>

namespace verona::rt
{
  template<typename T>
  class Noticeboard : public BaseNoticeboard
  {
  public:
    Noticeboard(T content_)
    {
      is_fundamental = std::is_fundamental_v<T>;
      put(content_);
    }

    void trace(ObjectStack& st) const
    {
      if constexpr (!std::is_fundamental_v<T>)
      {
#ifdef USE_SYSTEMATIC_TESTING_WEAK_NOTICEBOARDS
        for (auto p : update_buffer)
        {
          st.push((T)p);
        }
#endif
        auto p = get<T>();
        if (p)
          st.push(p);
      }
      else
      {
        UNUSED(st);
      }
    }

    // NOTE: the rc of new_o is not incremented
    void update(Alloc* alloc, T new_o)
    {
      if constexpr (!std::is_fundamental_v<T>)
      {
        assert(new_o->debug_is_immutable());
      }
#ifdef USE_SYSTEMATIC_TESTING_WEAK_NOTICEBOARDS
      update_buffer_push(new_o);
      flush_some(alloc);
      yield();
#else
      if constexpr (!std::is_fundamental_v<T>)
      {
        Epoch e(alloc);
        auto local_content = get<T>();
        Systematic::cout() << "Updating noticeboard " << this << " old value "
                           << local_content << " new value " << new_o
                           << Systematic::endl;
        e.dec_in_epoch(local_content);
        put(new_o);
      }
      else
      {
        UNUSED(alloc);
        put(new_o);
      }
#endif
    }

    T peek(Alloc* alloc)
    {
      if constexpr (std::is_fundamental_v<T>)
      {
        UNUSED(alloc);
        return get<T>();
      }
      else
      {
        T local_content;
        {
          // only protect incref with epoch
          Epoch e(alloc);
          local_content = get<T>();
          Systematic::cout() << "Inc ref from noticeboard peek" << local_content
                             << Systematic::endl;
          local_content->incref();
        }
        // It's possible that the following three things happen:
        // 1) cown is already Scanned,
        // 2) the owner of the noticeboard is in PreScan,
        // 3) the owner calls update
        // This way, the old content of the noticeboard is never scanned.
        // Intuitively, peek amounts to a way of receiving new msg, so it needs
        // to be scanned.
        if (Scheduler::should_scan())
        {
          Systematic::cout() << "Scan from noticeboard peek" << local_content
                             << Systematic::endl;
          ObjectStack f(alloc);
          local_content->trace(f);
          Cown::scan_stack(alloc, Scheduler::epoch(), f);
        }
        return local_content;
      }
    }
  };
} // namespace verona::rt
