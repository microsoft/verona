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
     * This represents a message that is sent to a behaviour.
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
      /**
       * TODO When we move to C++20, convert to returning a span.
       */
      Request* get_requests_array()
      {
        return snmalloc::pointer_offset<Request>(this, sizeof(Body));
      }

      Behaviour& get_behaviour()
      {
        return *snmalloc::pointer_offset<Behaviour>(
          this, sizeof(Body) + sizeof(Request) * count);
      }

      /**
       * Allocates a message body with sufficient space for the
       * cowns_array and the behaviour.  This does not initialise the cowns
       * array.
       */
      template<typename Be, typename... Args>
      static Body* make(Alloc& alloc, size_t count, Args... args)
      {
        size_t size = sizeof(Body) + (sizeof(Request) * count) + sizeof(Be);

        // Create behaviour
        auto body = new (alloc.alloc(size)) Body(count);
        new ((Be*)&(body->get_behaviour())) Be(std::forward<Args>(args)...);

        static_assert(
          alignof(Be) <= sizeof(void*), "Alignment not supported, yet!");
        static_assert(
          sizeof(Body) % alignof(Be) == 0, "Alignment not supported, yet!");

        return body;
      }
    };

  private:
    // The body of the actual message.
    // uses the bottom bit to determine if the request is a read.
    uintptr_t body_and_mode;
    friend verona::rt::MPSCQ<MultiMessage>;
    friend class Cown;
    friend struct Request;

    std::atomic<MultiMessage*> next{nullptr};

    inline Body* get_body()
    {
      auto result = (Body*)(body_and_mode & ~Object::MARK_MASK);
      return result;
    }

    static MultiMessage* make(Alloc& alloc, Body* body, bool is_read)
    {
      auto msg = (MultiMessage*)alloc.alloc<sizeof(MultiMessage)>();
      msg->body_and_mode = ((uintptr_t)body) | (is_read ? 1 : 0);
      return msg;
    }

    static MultiMessage* make_message(Alloc& alloc, Body* body, bool is_read)
    {
      MultiMessage* m = make(alloc, body, is_read);
      Logging::cout() << "MultiMessage " << m << " payload " << body
                      << Logging::endl;
      return m;
    }

    inline size_t size()
    {
      return sizeof(MultiMessage);
    }

    inline bool is_read()
    {
      return (body_and_mode & 1) == 1;
    }
  };
} // namespace verona::rt
