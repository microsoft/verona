// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../ds/mpscq.h"
#include "../object/object.h"
#include "behaviour.h"

#include <snmalloc/snmalloc.h>

namespace verona::rt
{
  using namespace snmalloc;

  class MultiMessage
  {
    /**
     * This represents a message that is send to a behaviour.
     *
     * The layout requires that after the allocation there is space for
     * `count` cown pointers, and then the behaviour's body.
     *
     * This layout allows the message body to be single allocation even though
     * there are multiple different sized pieces.
     */
    struct Body
    {
      std::atomic<size_t> exec_count_down;
      size_t count;

    private:
      Body(size_t count) : exec_count_down(count), count(count) {}

    public:
      Cown** get_cowns_array()
      {
        return snmalloc::pointer_offset<Cown*>(this, sizeof(Body));
      }

      Behaviour* get_behaviour()
      {
        return snmalloc::pointer_offset<Behaviour>(
          this, sizeof(Body) + sizeof(Cown*) * count);
      }

      /**
       * Allocates a message body with sufficient space for the
       * cowns_array and the behaviour.  Neither the cowns array
       * or the behaviour are initialised by this function.
       */
      static Body* make(Alloc& alloc, size_t count, size_t behaviour_size)
      {
        size_t size = sizeof(Body) + (sizeof(Cown*) * count) + behaviour_size;
        return new (alloc.alloc(size)) Body(count);
      }
    };

  private:
    // The body of the actual message.
    Body* body;
    friend verona::rt::MPSCQ<MultiMessage>;
    friend class Cown;

    std::atomic<MultiMessage*> next{nullptr};

    inline Body* get_body()
    {
      return (Body*)((uintptr_t)body & ~Object::MARK_MASK);
    }

    static MultiMessage* make(Alloc& alloc, EpochMark epoch, Body* body)
    {
      auto msg = (MultiMessage*)alloc.alloc<sizeof(MultiMessage)>();
      msg->body = body;
      msg->set_epoch(epoch);
      return msg;
    }

    inline bool in_epoch(EpochMark e)
    {
      return get_epoch() == e;
    }

    inline EpochMark get_epoch()
    {
      return (EpochMark)((uintptr_t)body & Object::MARK_MASK);
    }

    inline void set_epoch(EpochMark e)
    {
      assert(
        (e == EpochMark::EPOCH_NONE) || (e == EpochMark::EPOCH_A) ||
        (e == EpochMark::EPOCH_B));

      Logging::cout() << "MultiMessage epoch: " << this << " " << get_epoch()
                      << " -> " << e << Logging::endl;

      body = (Body*)((uintptr_t)get_body() | (size_t)e);

      assert(get_epoch() == e);
    }

    static MultiMessage* make_message(Alloc& alloc, Body* body, EpochMark epoch)
    {
      MultiMessage* m = make(alloc, epoch, body);
      Logging::cout() << "MultiMessage " << m << " payload " << body << " ("
                      << epoch << ")" << Logging::endl;
      return m;
    }

    inline size_t size()
    {
      return sizeof(MultiMessage);
    }
  };
} // namespace verona::rt
