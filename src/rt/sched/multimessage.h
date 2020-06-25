// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../ds/mpscq.h"
#include "../object/object.h"
#include "action.h"

#include <snmalloc.h>

namespace verona::rt
{
  using namespace snmalloc;

  class MultiMessage
  {
    struct MultiMessageBody
    {
      size_t index;
      size_t count;
      Cown** cowns;
      Action* action;
    };

  private:
    MultiMessageBody* body;
    friend verona::rt::MPSCQ<MultiMessage>;
    friend class Cown;

    std::atomic<MultiMessage*> next;

    inline MultiMessageBody* get_body()
    {
      return (MultiMessageBody*)((uintptr_t)body & ~Object::MARK_MASK);
    }

    static MultiMessage*
    make(Alloc* alloc, EpochMark epoch, MultiMessageBody* body)
    {
      auto msg = (MultiMessage*)alloc->alloc<sizeof(MultiMessage)>();
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

      Systematic::cout() << this << " state change " << get_epoch() << " -> "
                         << e << std::endl;

      body = (MultiMessageBody*)((uintptr_t)get_body() | (size_t)e);

      assert(get_epoch() == e);
    }

    static MultiMessageBody*
    make_body(Alloc* alloc, size_t count, Cown** cowns, Action* action)
    {
      auto body = (MultiMessageBody*)alloc->alloc<sizeof(MultiMessageBody)>();
      body->index = 0;
      body->count = count;
      body->cowns = cowns;
      body->action = action;

      Systematic::cout() << "MultiMessageBody " << body << std::endl;

      return body;
    }

    static MultiMessage*
    make_message(Alloc* alloc, MultiMessageBody* body, EpochMark epoch)
    {
      MultiMessage* m = make(alloc, epoch, body);
      Systematic::cout() << "MultiMessage " << m << " payload " << body << " ("
                         << epoch << ")" << std::endl;
      return m;
    }

    inline size_t size()
    {
      return sizeof(MultiMessage);
    }
  };
} // namespace verona::rt
